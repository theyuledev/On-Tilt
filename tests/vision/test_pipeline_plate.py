import json
import math

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision.pipeline import Pipeline
from src.vision.plate_find import SQUARE
from src.vision.px_to_mm import plate_px
from tests.unit.test_balls import ball
from tests.unit.test_plate_find import CALIB, at, paint

pytestmark = pytest.mark.unit

CFG = load()
VIS = {**CFG["vision"], "calib": CALIB}
FLAT = {**CFG, "vision": {**CFG["vision"], "px_per_mm": 2.0}}
FIXED = {**CFG, "vision": VIS}
TRACKED = {**CFG, "vision": {**VIS, "plate_track": True}}
HALF = CFG["rig"]["plate_inner_mm"] / 2
R_PX = round(CFG["rig"]["ball_d_mm"])  # 2 px per mm
REACH_PX = 640 + 2 * (HALF + CFG["rig"]["ball_d_mm"])
DT = 1 / 120


def table(dx=0.0, dy=0.0) -> np.ndarray:
    """The board with a light surface touching its +x side, the camera moved dx, dy px."""
    img = np.full((800, 1280), 60, np.uint8)
    cv2.rectangle(img, (int(900 + dx), int(150 + dy)), (1279, int(650 + dy)), 185, -1)
    return paint(img, at(SQUARE * HALF, dx, dy), 210)


ROLL_X = [800 + 10 * i for i in range(40)]


def rolled_off(cfg):
    """A ball rolling off the +x side and on across the light surface, 10 px a frame."""
    pipe = Pipeline(cfg)
    return [pipe.step(ball(table(), x, 400, R_PX), i * DT) for i, x in enumerate(ROLL_X)]


def test_default_has_no_plate_tracking_and_no_outline():
    pipe = Pipeline(CFG)
    o = pipe.step(table(), 0.0)
    assert pipe.plate is None and o.outline_px is None and o.occluded is False


def test_a_blob_off_a_tracked_board_is_not_the_ball():
    beside = ball(table(), 1130, 400, R_PX)
    assert not Pipeline(TRACKED).step(beside, 0.0).ball.valid
    # a fixed map can be off by more than a ball at a corner, so without tracking nothing is dropped
    assert Pipeline(FIXED).step(beside, 0.0).ball.valid
    assert Pipeline(FLAT).step(beside, 0.0).ball.valid
    both = ball(beside.copy(), 560, 330, R_PX)
    o = Pipeline(TRACKED).step(both, 0.0)
    assert o.ball.valid and (o.ball_px.x_px, o.ball_px.y_px) == pytest.approx((560, 330), abs=2)
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx((-40, 35), abs=1)


def test_nothing_is_dropped_while_the_board_is_not_found():
    # no board in the picture, so the corners are only the configured guess
    bare = ball(np.full((800, 1280), 185, np.uint8), 1130, 400, R_PX)
    o = Pipeline(TRACKED).step(bare, 0.0)
    assert o.occluded and o.ball.valid


def test_a_locked_ball_is_dropped_once_it_is_off_the_tracked_board():
    obs = rolled_off(TRACKED)
    assert not any(o.occluded for o in obs)
    on = [o.ball.valid for o, x in zip(obs, ROLL_X, strict=True) if x <= REACH_PX]
    off = [o.ball.valid for o, x in zip(obs, ROLL_X, strict=True) if x > REACH_PX]
    assert len(on) > 20 and all(on)
    # it coasts a few frames like any miss, then it is gone and stays gone
    assert len(off) > 10 and sum(off) < VIS["lost_after_misses"] and not any(off[-5:])
    assert all(o.ball.valid for o in rolled_off(FIXED))
    assert all(o.ball.valid for o in rolled_off(FLAT))


def test_tracked_plate_keeps_the_ball_mm_when_the_camera_moves():
    still = ball(table(), 700, 340, R_PX)
    moved = ball(table(7, -5), 707, 335, R_PX)
    tracked, fixed = Pipeline(TRACKED), Pipeline(FIXED)
    a, b = tracked.step(still, 0.0), tracked.step(moved, DT)
    assert not a.occluded and not b.occluded
    assert (a.ball.x_mm, a.ball.y_mm) == pytest.approx((30, 30), abs=0.3)
    assert math.dist((a.ball.x_mm, a.ball.y_mm), (b.ball.x_mm, b.ball.y_mm)) < 0.15
    # and no speed out of nothing: 4 mm in one frame would read as 500 mm/s
    assert math.hypot(b.ball.vx_mm_s, b.ball.vy_mm_s) < 20
    fixed.step(still, 0.0)
    c = fixed.step(moved, DT)
    assert math.dist((c.ball.x_mm, c.ball.y_mm), (30, 30)) > 4


def test_tracked_outline_follows_at_its_own_rate_and_only_when_the_board_moved():
    pipe = Pipeline(TRACKED)
    period = VIS["plate_outline_period_s"]
    first = pipe.step(table(), 0.0).outline_px
    assert first[0] == pytest.approx(plate_px(HALF, HALF, VIS), abs=0.3)
    # moved, but the outline is not due yet
    assert pipe.step(table(6, 4), period / 2).outline_px is first
    later = pipe.step(table(6, 4), period).outline_px
    assert later is not first
    assert np.array(later) - np.array(first) == pytest.approx(np.tile([6.0, 4.0], (48, 1)), abs=0.3)
    # due again, board where it was: the same list, so nothing is sent
    assert pipe.step(table(6, 4), 2 * period).outline_px is later


def test_no_board_is_occluded_and_the_map_stays():
    pipe = Pipeline(TRACKED)
    seen = pipe.step(ball(table(), 700, 340, R_PX), 0.0)
    period = VIS["plate_outline_period_s"]
    dark = pipe.step(np.full((800, 1280), 60, np.uint8), period)
    assert dark.occluded is True and dark.outline_px is seen.outline_px
    assert json.dumps(dark.occluded) == "true"
    back = pipe.step(ball(table(), 700, 340, R_PX), 2 * period)
    assert back.occluded is False
    assert (back.ball.x_mm, back.ball.y_mm) == pytest.approx((seen.ball.x_mm, seen.ball.y_mm))
