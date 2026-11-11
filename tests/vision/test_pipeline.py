import math

import numpy as np
import pytest

from src.io.settings import ROOT, load
from src.vision import calib
from src.vision.pipeline import Pipeline
from tests.unit.test_balls import ball, plate

pytestmark = pytest.mark.unit

CFG = load()
VIS = CFG["vision"]
DT = 1 / 120
SPAN = VIS["velocity_span_frames"]
RIG = calib.load_calib(ROOT / "tests/fixtures/lens_rig.json")


def rolling(n, start=0):
    # 10 px right and 10 px down the image every frame: 1200 px/s
    return [
        (ball(plate(), 640 + 10 * i, 400 + 10 * i, 40), i * DT) for i in range(start, start + n)
    ]


def run(pipe, frames):
    return [pipe.step(image, t) for image, t in frames]


def test_ball_time_is_the_read_time_and_the_velocity_spans_the_frames():
    obs = run(Pipeline(CFG), rolling(SPAN + 1))
    assert all(o.ball.valid and o.ball.t_s == o.t_read for o in obs)
    assert obs[-1].t_read == SPAN * DT
    speed = 1200 / VIS["px_per_mm"]
    assert obs[-1].ball.vx_mm_s == pytest.approx(speed, rel=0.1)
    assert obs[-1].ball.vy_mm_s == pytest.approx(-speed, rel=0.1)


def test_same_frame_twice_is_the_same_observation():
    pipe = Pipeline(CFG)
    frames = rolling(3)
    first = run(pipe, frames)[-1]
    assert pipe.step(*frames[-1]) is first
    assert pipe.latest() is first


def test_relock_after_a_loss_starts_the_velocity_from_zero():
    pipe = Pipeline(CFG)
    n = SPAN + 1
    run(pipe, rolling(n))
    empty = plate()
    lost = [pipe.step(empty, (n + i) * DT) for i in range(VIS["lost_after_misses"])]
    assert not lost[-1].ball.valid
    back = run(pipe, rolling(1, start=n + VIS["lost_after_misses"]))[0]
    assert back.ball.valid and (back.ball.vx_mm_s, back.ball.vy_mm_s) == (0.0, 0.0)


def test_tilt_is_nan_and_nothing_occluded_without_markers():
    o = run(Pipeline(CFG), rolling(1))[0]
    assert math.isnan(o.tilt_x_deg) and math.isnan(o.tilt_y_deg)
    assert o.occluded is False
    assert o.ms > 0


def test_a_lens_undistorts_the_ball_and_no_lens_keeps_the_flat_map():
    # reads no file: module scoped fixtures build pipelines before any per test guard runs
    flat = Pipeline(CFG)
    assert flat.lens is None
    frame = [(ball(plate(), 1000, 650, 40), 0.0)]
    a, b = run(flat, frame)[0].ball, run(Pipeline(CFG, RIG), frame)[0].ball
    assert math.hypot(b.x_mm, b.y_mm) > 1.05 * math.hypot(a.x_mm, a.y_mm)


def raw_px(lens, ideal):
    # where the lens puts undistorted px, straight from its forward model
    K = np.array(lens["K"])
    n = (np.asarray(ideal, float) - K[:2, 2]) / K.diagonal()[:2]
    return calib.project(lens, np.column_stack([n, np.ones(len(n))]))


def test_rig_lens_puts_a_corner_ball_back_out_where_it_is():
    s = VIS["px_per_mm"]
    # plate centre at (640, 400) and the ball near the top left corner, both undistorted px
    centre, at = raw_px(RIG, [(640, 400), (300, 120)])
    cfg = {**CFG, "vision": {**VIS, "plate_centre_px": list(centre), "plate_yaw_deg": 0.0}}
    frame = [(ball(plate(), *np.round(at).astype(int), 40), 0.0)]
    flat = run(Pipeline(cfg), frame)[0].ball
    bent = run(Pipeline(cfg, RIG), frame)[0].ball
    assert (bent.x_mm, bent.y_mm) == pytest.approx((-340 / s, 280 / s), rel=0.02)
    factor = math.hypot(340, 280) / math.dist(at, centre)
    assert math.hypot(bent.x_mm, bent.y_mm) / math.hypot(flat.x_mm, flat.y_mm) == pytest.approx(
        factor, rel=0.02
    )


def test_no_lens_is_main_flat_map_to_the_bit():
    o = run(Pipeline(CFG), [(ball(plate(), 1000, 650, 40), 0.0)])[0]
    (cx, cy), s = VIS["plate_centre_px"], VIS["px_per_mm"]
    assert VIS["plate_yaw_deg"] == 0.0
    assert (o.ball.x_mm, o.ball.y_mm) == ((o.ball_px.x_px - cx) / s, (cy - o.ball_px.y_px) / s)
