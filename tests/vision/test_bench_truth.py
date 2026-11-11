import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.vision.bench import rec, truth
from src.vision.bench.track import ARRIVED, NEVER, NO_TRAVEL, Arrivals, Track
from src.vision.bench.truth import (
    ALONE,
    ARRIVAL,
    DISPUTED,
    SECOND,
    STILL,
    UNREADABLE,
    UNSEEN,
    Label,
    describe,
    labels,
)
from tools import replay_clip

pytestmark = pytest.mark.unit

BENCH = {"rest_mm_s": 5.0, "rest_mm": 3.0, "rest_s": 1.0}
A, B = (100.0, 100.0), (140.0, 100.0)
R = 20.0
GONE = {"valid": False, "px": None}


def snap(t_s: float, mm=(0.0, 0.0), px=A, v: float = 0.0, valid: bool = True) -> dict:
    ball = {"x_mm": mm[0], "y_mm": mm[1], "vx_mm_s": v, "vy_mm_s": 0.0, "valid": valid}
    return {"t_s": t_s, "armed": True, "ball": ball, "ball_px": [*px, R] if px else None}


def shot(t_s: float = 100.0, stretch: int = 0, b_px=None, **ball) -> rec.Shot:
    """A picture between two snapshots 50 ms apart, the ball at b_px by the second one."""
    after = snap(t_s + 0.05, **{**ball, "px": b_px or ball.get("px", A)})
    return rec.Shot(Path(f"{t_s:010.3f}.jpg"), snap(t_s, **ball), after, stretch)


def lying(start: float = 90.0, end: float = 110.0) -> Track:
    t = np.arange(start, end, 0.01)
    return Track(t, np.zeros((len(t), 2)))


def far(_shot) -> tuple[float, float]:
    return 300.0, 300.0


def test_a_frame_on_a_track_that_arrived_is_labelled():
    assert labels([shot(b_px=B)], [lying()], {0}, BENCH) == [Label(A, B, R, ARRIVAL, True, None)]


def test_a_frame_without_support_gets_no_label_and_a_reason():
    assert labels([shot()], [lying()], set(), BENCH) == [ALONE]
    # the two snapshots sit on different tracks: nothing carries the arrival across
    cut = [lying(90.0, 100.02), lying(100.03, 110.0)]
    assert labels([shot()], cut, {0, 1}, BENCH) == [ALONE]
    assert labels([shot(**GONE)], [lying()], {0}, BENCH) == [UNSEEN]
    assert labels([shot(px=None)], [lying()], {0}, BENCH) == [UNSEEN]


def test_a_moving_balls_label_is_the_path_between_snapshots():
    moving = shot(b_px=B, v=800.0)
    # 15 px off the middle of the path, 25 px from either end
    got = labels([moving], [], set(), BENCH, lambda s: (120.0, 115.0))
    assert got == [Label(A, B, R, SECOND, False, True)]
    assert labels([moving], [], set(), BENCH, lambda s: (120.0, 121.0)) == [DISPUTED]
    assert labels([moving], [], set(), BENCH, lambda s: (161.0, 100.0)) == [DISPUTED]
    assert labels([moving], [], set(), BENCH, lambda s: (79.0, 100.0)) == [DISPUTED]


def test_a_disagreement_is_disputed_not_wrong():
    # with an arrival behind it the label stands and the dispute is written down
    assert labels([shot()], [lying()], {0}, BENCH, far) == [Label(A, A, R, ARRIVAL, True, False)]
    # with nothing else behind it there is no label, and no verdict either
    assert labels([shot()], [lying()], set(), BENCH, far) == [DISPUTED]


def test_an_unreadable_picture_is_skipped_with_its_reason():
    assert labels([shot()], [lying()], {0}, BENCH, lambda s: None) == [UNREADABLE]


def test_at_rest_needs_a_slow_ball_that_stayed_put():
    def rest(one: rec.Shot, track: Track) -> bool:
        return labels([one], [track], {0}, BENCH)[0].at_rest

    assert rest(shot(), lying())
    assert not rest(shot(v=5.0), lying())
    # 3 mm/s: 3.15 mm over the second before the picture and its 50 ms
    drift = lying()
    drift.xy[:, 0] = np.linspace(0.0, 60.0, len(drift.t))
    assert not rest(shot(), drift)
    # seen for half a second only
    assert not rest(shot(), lying(start=99.5))


def test_a_still_ball_between_agreeing_sightings_is_labelled_while_unseen():
    seen = [lying(90.0, 100.2), lying(102.9, 110.0)]

    def middle(arrived=(0, 1), last_mm=(1.0, 0.0), **first) -> list:
        shots = [shot(100.0, **first), shot(101.0, **GONE), shot(102.0, **GONE)]
        last = shot(104.0, mm=last_mm, px=(101.0, 100.0))
        return labels([*shots, last], seen, set(arrived), BENCH)[1:3]

    # at the place it was last seen
    still = Label(A, A, R, STILL, True, None)
    assert middle() == [still, still]
    # afterwards it lies somewhere else
    assert middle(last_mm=(3.0, 0.0)) == [UNSEEN, UNSEEN]
    # it was rolling when last seen
    assert middle(v=5.0) == [UNSEEN, UNSEEN]
    # the sighting before has nothing behind it
    assert middle(arrived=(1,)) == [UNSEEN, UNSEEN]
    apart = [shot(100.0), shot(101.0, **GONE), shot(104.0, stretch=1)]
    assert labels(apart, seen, {0, 1}, BENCH)[1] == UNSEEN


def test_describe_counts_labels_by_support_and_every_skip_by_reason():
    got = [
        Label(A, A, R, ARRIVAL, True, True),
        Label(A, B, R, ARRIVAL, False, False),
        Label(A, A, R, SECOND, True, True),
        UNSEEN,
        UNSEEN,
        DISPUTED,
    ]
    recording = rec.Recording([], 7, Counter({rec.NO_NEXT: 1}), 1, Counter())
    arrived = Arrivals(set(), Counter({ARRIVED: 3, NEVER: 1}), 4)
    assert describe("one", recording, got, arrived).splitlines() == [
        f"one: 3 of 7 frames labelled (2 {ARRIVAL}, 1 {SECOND}), 2 at rest",
        f"  second detector agrees on 1 of 2 frames that {ARRIVAL}",
        f"  4 trials: 3 {ARRIVED}, 1 {NEVER}; stuck not recorded in 4",
        f"  skipped 4 of 7 frames: 2 {UNSEEN}, 1 {rec.NO_NEXT}, 1 {DISPUTED}",
    ]
    empty = rec.Recording([], 0, Counter(), 0, Counter())
    assert describe("two", empty, [], Arrivals(set(), Counter(), 0)).splitlines() == [
        "two: 0 of 0 frames labelled, 0 at rest",
        "  skipped 0 of 0 frames",
    ]


def picture(path: Path, x: int, y: int) -> None:
    grain = np.random.default_rng(1).integers(-4, 5, (300, 400))
    img = (200 + grain).astype(np.uint8)
    cv2.circle(img, (x, y), 19, 30, -1)
    cv2.imwrite(str(path), img)


def recorded(tmp_path: Path) -> Path:
    """A recorder folder: the ball rolls to a driver target and lies there in two pictures."""
    folder = tmp_path / "one"
    (folder / "frames").mkdir(parents=True)
    (folder / "settings.local.toml").write_text("[rig]\nball_d_mm = { value = 16 }\n")
    at = (200.0, 150.0)
    snaps = [snap(100.0, px=at), snap(100.05, px=at), snap(100.1, px=at), snap(100.15, px=at)]
    for row in snaps:
        row["ball_px"][2] = 19.0
    # rolls in from 60 mm out at 40 mm/s, then lies on the target
    times = np.arange(95.0, 100.3, 0.01).tolist()
    xs = [min(-60 + 40 * (t - 95.0), 0.0) for t in times]
    log = [
        {"t_s": t, "ball": {"x_mm": x, "y_mm": 0.0, "valid": True, "t_s": t}}
        for t, x in zip(times, xs, strict=True)
    ]
    trials = [
        {"target": [0, 0], "t_start": 95.0, "reached_s": 1.5, "ball_lost": False},
        # the next target is 10 mm from where it lies: no travel to speak of
        {"target": [10, 0], "t_start": 99.0, "reached_s": 0.01, "ball_lost": False},
        {"ended": "time is up"},
    ]
    for name, rows in ("snap", snaps), ("run", log), ("trials", trials):
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    picture(folder / "frames" / "000100.000.jpg", 200, 150)
    # the second picture shows the ball somewhere the recorded position is not
    picture(folder / "frames" / "000100.100.jpg", 80, 80)
    return folder


def test_build_cuts_tracks_at_the_bench_top_speed_and_the_coast_time(tmp_path, monkeypatch):
    asked = {}

    def spy(rows, top_mm_s, hole_s):
        asked.update(top_mm_s=top_mm_s, hole_s=hole_s)
        return []

    monkeypatch.setattr(truth, "tracks", spy)
    folder = recorded(tmp_path)
    cfg = rec.settings(folder)
    truth.build(folder, cfg)
    coast_s = cfg["vision"]["lost_after_misses"] / cfg["rig"]["camera_fps"]
    assert asked == {"top_mm_s": cfg["bench"]["top_speed_mm_s"], "hole_s": coast_s}


def test_the_labels_command_writes_a_row_a_picture_and_what_it_skipped(tmp_path, capsys):
    folder = recorded(tmp_path)

    replay_clip.main(["--labels", str(tmp_path / "truth"), str(folder)])

    assert capsys.readouterr().out.splitlines() == [
        f"one: 2 of 2 frames labelled (2 {ARRIVAL}), 2 at rest",
        f"  second detector agrees on 1 of 2 frames that {ARRIVAL}",
        f"  2 trials: 1 {ARRIVED}, 1 {NO_TRAVEL}; stuck not recorded in 2",
        "  skipped 0 of 2 frames",
    ]
    rows = [json.loads(x) for x in (tmp_path / "truth" / "one.jsonl").read_text().splitlines()]
    label = {"a": [200.0, 150.0], "b": [200.0, 150.0], "r_px": 19.0, "support": ARRIVAL}
    assert rows == [
        {"frame": "000100.000.jpg", **label, "at_rest": True, "second": True},
        {"frame": "000100.100.jpg", **label, "at_rest": True, "second": False},
    ]
