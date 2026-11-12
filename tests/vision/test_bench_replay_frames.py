import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision.bench import rec, replay
from src.vision.bench.replay import MOVING, NO_BOARD, NO_ROW, WARMING, Fed
from src.vision.bench.score import ALL, CORRECT, MIDDLE
from src.vision.bench.truth import UNREADABLE, UNSEEN
from src.vision.plate_find import SQUARE
from tests.unit.test_balls import ball
from tests.unit.test_plate_find import CALIB, at, paint
from tests.vision.test_bench_replay import (
    FLAT,
    MID,
    WARM,
    R,
    lying,
    name,
    planted,
    recorded,
    run,
    verdicts,
)
from tools import replay_clip

pytestmark = pytest.mark.unit


def test_the_tracker_gets_the_picture_in_colour_as_it_does_from_the_camera(tmp_path, monkeypatch):
    shapes = []
    step = replay.Pipeline.step

    def spy(self, image, t_s):
        shapes.append(image.shape)
        return step(self, image, t_s)

    monkeypatch.setattr(replay.Pipeline, "step", spy)
    run(recorded(tmp_path, [MID]), lying([MID]))
    assert shapes == [(300, 400, 3)]


def test_background_is_fed_from_truth():
    cfg = load()["vision"]
    plate = np.full((300, 400), 200, np.uint8)
    with_ball = ball(plate.copy(), *MID, R)
    fed = Fed(cfg)
    fed.keep = (MID[0] - 2 * R, MID[1] - 2 * R, MID[0] + 2 * R, MID[1] + 2 * R)
    # the tracker saw nothing and passes no box: the ball stays out all the same
    fed.update(with_ball, 0.0, None, True)
    assert len(fed.samples) == 1 and fed.samples[0].min() > 150
    # and its own answer somewhere else is not what is kept out
    fed.keep = (0, 0, 40, 40)
    fed.update(with_ball, 10.0, fed.keep, True)
    assert len(fed.samples) == 2 and fed.samples[1].min() < 100


def test_a_frame_without_a_label_is_never_sampled_into_the_background(tmp_path):
    fed = Fed(load()["vision"])
    fed.update(np.full((300, 400), 200, np.uint8), 0.0, (0, 0, 40, 40), True)
    assert fed.samples == []
    # four pictures the live run did not see the ball in, then eight labelled ones: four samples
    # at most, so no picture is scored. sampled all along it would be warm for the last three
    spots = [None] * 4 + [MID] * (WARM - 1)
    got = run(recorded(tmp_path, spots), lying(spots))
    assert got.cold == [] and got.skipped == Counter({WARMING: WARM - 1, UNSEEN: 4})


def test_warm_up_frames_are_skipped_and_counted_again_after_a_gap(tmp_path):
    spots = [MID] * (WARM + 1)
    recorded(tmp_path, spots)
    folder = recorded(tmp_path, spots, t0=500.0)
    got = run(folder, {**lying(spots), **lying(spots, t0=500.0)})
    assert got.frames == 2 * (WARM + 1) and got.skipped == Counter({WARMING: 2 * WARM})
    assert verdicts(got.cold) == [CORRECT] * 2


def test_pictures_that_cannot_be_scored_keep_their_reason(tmp_path):
    spots = [MID] * (WARM + 3)
    folder = recorded(tmp_path, spots)
    labels = lying(spots)
    del labels[name(100.0 + WARM)]
    (folder / "frames" / name(101.0 + WARM)).write_bytes(b"")
    (folder / "frames" / "000001.000.jpg").write_bytes(b"")
    got = run(folder, labels)
    assert got.frames == WARM + 4 and verdicts(got.cold) == [CORRECT]
    assert got.skipped == Counter({WARMING: WARM, NO_ROW: 1, UNREADABLE: 1, rec.NO_STAMP: 1})
    assert len(got.cold) + sum(got.skipped.values()) == got.frames


TRACKED = (
    "[rig]\nball_d_mm = { value = 16 }\nplate_inner_mm = { value = 240 }\n"
    "[vision]\nuse_calib = true\nplate_track = true\n"
)
HALF = 120.0


def board(path: Path, spot: tuple[int, int], dx: float | None) -> None:
    """The full-size picture of a 240 mm board at 2 px a mm, the camera moved dx px. None: no
    board in the picture at all."""
    img = np.full((800, 1280), 60 if dx is not None else 185, np.uint8)
    if dx is not None:
        img = paint(img, at(SQUARE * HALF, dx), 210)
    cv2.imwrite(str(path), ball(img, *spot, R))


def tracked(tmp_path: Path, spot: tuple[int, int], dx: float | None) -> replay.Scored:
    spots = [spot] * (WARM + 1)
    folder = recorded(tmp_path, spots, TRACKED)
    for i in range(len(spots)):
        board(folder / "frames" / name(100.0 + i), spot, dx)
    cfg = rec.calibrated(rec.settings(folder), CALIB)
    return replay.run(rec.read(folder, cfg["vision"]["bg_period_s"]), lying(spots), cfg)


def test_the_region_comes_from_the_board_as_the_picture_shows_it(tmp_path):
    # the camera moved 30 px: 35 mm right of the board's middle, 50 mm right of where the
    # settings put it. a third of this board is 40 mm
    got = tracked(tmp_path, (640 + 30 + 70, 400), 30.0)
    assert [m.regions for m in got.cold] == [(MIDDLE, ALL)]
    assert verdicts(got.cold) == verdicts(got.locked) == [CORRECT]
    # measured on that board too, not 15 mm off on the one in the settings
    assert all(m.off_mm < 1.0 for m in got.cold + got.locked)


def test_a_picture_where_the_board_is_not_found_is_skipped_and_counted(tmp_path):
    got = tracked(tmp_path, (640, 400), None)
    assert got.cold == [] and got.locked == []
    assert got.skipped == Counter({WARMING: WARM, NO_BOARD: 1})


def label_file(path: Path, labels: dict) -> None:
    rows = [{"frame": f, **got._asdict()} for f, got in labels.items()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_score_command_prints_each_folder_and_the_total(tmp_path, capsys, monkeypatch):
    planted(monkeypatch, (MID[0] + 2, MID[1]))
    spots = [MID] * (WARM + 2)
    # on the calibration handed in, which puts the middle of this board at -220, 125 mm
    local = FLAT + "use_calib = true\ncalib_centre_mm = { value = [-220, 125] }\n"
    one, two = (recorded(tmp_path, spots, local, called=n) for n in ("one", "two"))
    truth = tmp_path / "truth"
    truth.mkdir()
    label_file(truth / "one.jsonl", lying(spots))
    label_file(truth / "two.jsonl", lying(spots, at_rest=False))
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps(CALIB))

    replay_clip.main(["--score", str(truth), str(calib), str(one)])
    alone = capsys.readouterr().out.splitlines()
    assert alone[:3] == [
        f"one: 2 of {WARM + 2} frames scored",
        f"  skipped {WARM} of {WARM + 2} frames: {WARM} {WARMING}",
        "  found from nothing",
    ]
    row = "2      2 100.0%      0   0.0%      0   0.0%  1.00, 1.00"
    assert (
        alone[4] == f"    middle             {row}" and alone[9] == f"    all                {row}"
    )
    assert alone[10] == "  one step locked on a ball at rest, not run on 0 scored frames"
    assert alone[17] == f"    all                {row}" and len(alone) == 18

    replay_clip.main(["--score", str(truth), str(calib), str(one), str(two)])
    both = capsys.readouterr().out.splitlines()
    assert both[:18] == alone and len(both) == 54
    moving = "2      2 100.0%      0   0.0%      0   0.0%  -"
    assert both[18] == f"two: 2 of {WARM + 2} frames scored"
    assert both[27] == f"    all                {moving}"
    assert (
        both[28] == f"  one step locked on a ball at rest, not run on 2 scored frames: 2 {MOVING}"
    )
    assert both[35] == "    all                0      0      -      0      -      0      -  -"
    assert both[36:38] == [
        f"all 2 folders: 4 of {2 * WARM + 4} frames scored",
        f"  skipped {2 * WARM} of {2 * WARM + 4} frames: {2 * WARM} {WARMING}",
    ]
    assert (
        both[45] == "    all                4      4 100.0%      0   0.0%      0   0.0%  1.00, 1.00"
    )
    assert both[46] == both[28] and both[53] == alone[17]
