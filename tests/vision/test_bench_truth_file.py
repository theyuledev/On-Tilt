import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.bench import rec, truth_file
from src.vision.bench.track import ARRIVED, BROKEN, NEVER, NO_TRAVEL, Arrivals
from src.vision.bench.truth import ARRIVAL, DISPUTED, SECOND, UNREADABLE, UNSEEN, Label
from tools import replay_clip

pytestmark = pytest.mark.unit

A, B = (100.0, 100.0), (140.0, 100.0)


def test_describe_counts_labels_by_support_and_every_skip_by_reason():
    got = [
        Label(A, A, 20.0, ARRIVAL, True, True),
        Label(A, B, 20.0, ARRIVAL, False, False),
        Label(A, A, 20.0, SECOND, True, True),
        UNSEEN,
        UNSEEN,
        DISPUTED,
    ]
    recording = rec.Recording([], 7, Counter({rec.NO_NEXT: 1}), 1, Counter())
    arrived = Arrivals(set(), Counter({ARRIVED: 3, NEVER: 1}), 4)
    built = truth_file.Built(recording, got, arrived, 2)
    assert truth_file.describe("one", built).splitlines() == [
        f"one: 3 of 7 frames labelled (2 {ARRIVAL}, 1 {SECOND}), 2 at rest",
        f"  second detector agrees on 1 of 2 frames that {ARRIVAL}",
        f"  4 trials: 3 {ARRIVED}, 1 {NEVER}; stuck not recorded in 4",
        "  2 torn log lines left out",
        f"  skipped 4 of 7 frames: 2 {UNSEEN}, 1 {rec.NO_NEXT}, 1 {DISPUTED}",
    ]
    empty = rec.Recording([], 0, Counter(), 0, Counter())
    bare = truth_file.Built(empty, [], Arrivals(set(), Counter(), 0), 0)
    assert truth_file.describe("two", bare).splitlines() == [
        "two: 0 of 0 frames labelled, 0 at rest",
        "  skipped 0 of 0 frames",
    ]


def picture(path: Path, *discs: tuple[int, int, int]) -> None:
    grain = np.random.default_rng(1).integers(-4, 5, (300, 400))
    img = (200 + grain).astype(np.uint8)
    for x, y, r in discs:
        cv2.circle(img, (x, y), r, 30, -1)
    cv2.imwrite(str(path), img)


def snap(t_s: float, seen: bool = True) -> dict:
    ball = {"x_mm": 0.0, "y_mm": 0.0, "vx_mm_s": 0.0, "vy_mm_s": 0.0, "valid": seen}
    px = [200.0, 150.0, 19.0] if seen else None
    return {"t_s": t_s, "armed": True, "ball": ball, "ball_px": px}


def recorded(tmp_path: Path) -> Path:
    """A recorder folder with a 16 mm ball: it rolls to a driver target, lies there in two
    pictures and is lost before the third."""
    folder = tmp_path / "one"
    (folder / "frames").mkdir(parents=True)
    (folder / "settings.local.toml").write_text("[rig]\nball_d_mm = { value = 16 }\n")
    times = [100.0, 100.05, 100.1, 100.15]
    snaps = [*map(snap, times), snap(100.2, False), snap(100.25, False)]
    # in from 60 mm out at 40 mm/s, then on the target; the log loses it at 100.17
    log = []
    for t in np.arange(95.0, 100.3, 0.01).tolist():
        x = min(-60 + 40 * (t - 95.0), 0.0)
        ball = {"x_mm": x, "y_mm": 0.0, "valid": t < 100.17, "t_s": t}
        log.append({"t_s": t, "ball": ball})
    trials = [
        {"target": [0, 0], "t_start": 95.0, "reached_s": 1.5, "ball_lost": False},
        # the next target is 10 mm from where it lies: no travel to speak of
        {"target": [10, 0], "t_start": 99.0, "reached_s": 0.01, "ball_lost": False},
        {"ended": "time is up"},
    ]
    for name, rows in ("snap", snaps), ("run", log), ("trials", trials):
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with (folder / "run.jsonl").open("a") as f:
        f.write('{"t_s": 100.3, "pla')
    # a dark disc twice the ball's size lies beside the ball
    picture(folder / "frames" / "000100.000.jpg", (200, 150, 19), (310, 230, 38))
    # the ball somewhere the recorded position is not
    picture(folder / "frames" / "000100.100.jpg", (80, 80, 19))
    picture(folder / "frames" / "000100.200.jpg")
    return folder


def test_build_cuts_tracks_at_the_bench_top_speed_and_the_coast_time(tmp_path, monkeypatch):
    asked = {}

    def spy(rows, top_mm_s, hole_s):
        asked.update(top_mm_s=top_mm_s, hole_s=hole_s)
        return []

    monkeypatch.setattr(truth_file, "tracks", spy)
    folder = recorded(tmp_path)
    cfg = rec.settings(folder)
    truth_file.build(folder, cfg)
    coast_s = cfg["vision"]["lost_after_misses"] / cfg["rig"]["camera_fps"]
    assert asked == {"top_mm_s": cfg["bench"]["top_speed_mm_s"], "hole_s": coast_s}


def by_disc(folder: Path) -> truth_file.Built:
    cfg = rec.settings(folder)
    return truth_file.build(folder, cfg, truth_file.by_disc(cfg))


def test_a_folder_without_trials_is_labelled_by_the_second_detector_alone(tmp_path):
    folder = recorded(tmp_path)
    (folder / "trials.jsonl").unlink()
    built = by_disc(folder)
    got = [g if isinstance(g, str) else g.support for g in built.labels]
    assert got == [SECOND, DISPUTED, UNSEEN]
    assert not built.arrived.trials


def test_a_folder_without_a_run_log_is_labelled_by_the_second_detector_alone(tmp_path):
    folder = recorded(tmp_path)
    (folder / "run.jsonl").unlink()
    built = by_disc(folder)
    got = [g if isinstance(g, str) else (g.support, g.at_rest) for g in built.labels]
    # no log, so no track: nothing arrived anywhere and nothing is known to lie at rest
    assert got == [(SECOND, False), DISPUTED, UNSEEN]
    assert built.arrived.trials == Counter({BROKEN: 2}) and built.torn == 0


def test_read_gives_back_what_write_wrote(tmp_path):
    built = by_disc(recorded(tmp_path))
    moving = Label((1.5, 2.0), (3.0, 4.5), 19.0, ARRIVAL, False, None)
    built = built._replace(labels=[moving, *built.labels[1:]])
    truth_file.write(tmp_path / "one.jsonl", built)
    names = [shot.path.name for shot in built.recording.shots]
    got = truth_file.read(tmp_path / "one.jsonl")
    assert got == dict(zip(names, built.labels, strict=True))
    assert [type(g) for g in got.values()] == [Label, Label, str]


def test_a_picture_that_will_not_open_is_skipped_as_unreadable(tmp_path):
    folder = recorded(tmp_path)
    (folder / "frames" / "000100.000.jpg").write_bytes(b"")
    assert by_disc(folder).labels[0] == UNREADABLE


def test_the_labels_command_writes_a_row_a_picture_and_what_it_skipped(tmp_path, capsys):
    folder = recorded(tmp_path)

    replay_clip.main(["--labels", str(tmp_path / "truth"), str(folder)])

    assert capsys.readouterr().out.splitlines() == [
        f"one: 2 of 3 frames labelled (2 {ARRIVAL}), 2 at rest",
        f"  second detector agrees on 1 of 2 frames that {ARRIVAL}",
        f"  2 trials: 1 {ARRIVED}, 1 {NO_TRAVEL}; stuck not recorded in 2",
        "  1 torn log lines left out",
        f"  skipped 1 of 3 frames: 1 {UNSEEN}",
    ]
    rows = [json.loads(x) for x in (tmp_path / "truth" / "one.jsonl").read_text().splitlines()]
    label = {"a": [200.0, 150.0], "b": [200.0, 150.0], "r_px": 19.0, "support": ARRIVAL}
    assert rows == [
        {"frame": "000100.000.jpg", **label, "at_rest": True, "second": True},
        {"frame": "000100.100.jpg", **label, "at_rest": True, "second": False},
        {"frame": "000100.200.jpg", "skip": UNSEEN},
    ]
