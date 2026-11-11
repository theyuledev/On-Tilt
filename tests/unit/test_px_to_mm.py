import math

import numpy as np
import pytest

from src.contracts import BallState
from src.vision import calib
from src.vision.px_to_mm import ball_state
from src.vision.tracker import BallPx

pytestmark = pytest.mark.unit

CFG = {
    "px_per_mm": 2.0,
    "plate_centre_px": [640, 400],
    "plate_yaw_deg": 0.0,
    "velocity_span_frames": 4,
}
SPAN = CFG["velocity_span_frames"]


def seen(x, y, t, valid=True):
    return BallPx(x, y, 40.0, valid, t)


def track(points, dt=0.01, cfg=CFG, lens=None):
    states = []
    for i, (x, y) in enumerate(points):
        states.append(ball_state(seen(x, y, i * dt), states[-SPAN:], cfg, lens))
    return states


def test_plate_centre_is_the_origin_with_y_up_and_the_scale_in_mm():
    assert ball_state(seen(640, 400, 1.0), [], CFG) == BallState(0.0, 0.0, 0.0, 0.0, True, 1.0)
    s = ball_state(seen(660, 380, 1.0), [], CFG)
    assert (s.x_mm, s.y_mm) == (10.0, 10.0)
    s = ball_state(seen(600, 440, 1.0), [], CFG)
    assert (s.x_mm, s.y_mm) == (-20.0, -20.0)


def test_plate_yaw_turns_the_flipped_image_axes_onto_the_plate():
    cfg = {**CFG, "plate_yaw_deg": 90.0}
    right = ball_state(seen(660, 400, 1.0), [], cfg)
    assert (right.x_mm, right.y_mm) == pytest.approx((0.0, 10.0))
    up = ball_state(seen(640, 380, 1.0), [], cfg)
    assert (up.x_mm, up.y_mm) == pytest.approx((-10.0, 0.0))


def test_velocity_is_the_difference_over_four_frames():
    # 2 px right and 2 px down the image every 10 ms: +100 mm/s in x, -100 in y
    states = track([(640 + 2 * i, 400 + 2 * i) for i in range(10)])
    assert (states[0].vx_mm_s, states[0].vy_mm_s) == (0.0, 0.0)
    assert states[-1].vx_mm_s == pytest.approx(100) and states[-1].vy_mm_s == pytest.approx(-100)

    # stopped for one frame: the last frame alone says 0, four frames back say 6 px in 40 ms
    stop = track([(640, 400), (642, 400), (644, 400), (646, 400), (648, 400), (648, 400)])
    assert stop[-1].vx_mm_s == pytest.approx(75)


def test_lost_ball_is_invalid_and_velocity_restarts_from_the_last_good_fix():
    states = track([(640, 400), (650, 400)])
    lost = ball_state(seen(0, 0, 0.02, valid=False), states, CFG)
    assert not lost.valid and (lost.vx_mm_s, lost.vy_mm_s) == (0.0, 0.0)
    back = ball_state(seen(660, 400, 0.03), [*states, lost], CFG)
    assert back.valid and back.vx_mm_s == pytest.approx(10 / 0.03)


# wide lens like the overhead cam, looking straight down from 350 mm: 2 px/mm on the plate
LENS = {
    "model": "standard",
    "K": [[700.0, 0, 640], [0, 700.0, 400], [0, 0, 1]],
    "D": [-0.3, 0.1, 0, 0, -0.02],
}
HEIGHT = 350.0
RIM = 150.0
GRID = [(x, y) for x in np.linspace(-RIM, RIM, 7) for y in np.linspace(-RIM, RIM, 7)]


def from_above(points, off=(0.0, 0.0)):
    # plate y is up the image, camera y is down it
    return calib.project(LENS, [(off[0] + x, off[1] - y, HEIGHT) for x, y in points])


@pytest.mark.parametrize("off", [(0.0, 0.0), (-40.0, 60.0)])
def test_lens_puts_the_ball_back_where_it_is_out_to_the_rim(off):
    cfg = {**CFG, "plate_centre_px": list(from_above([(0, 0)], off)[0])}
    for (x, y), (u, v) in zip(GRID, from_above(GRID, off), strict=True):
        s = ball_state(seen(u, v, 1.0), [], cfg, LENS)
        assert math.hypot(s.x_mm - x, s.y_mm - y) < 1e-3 * RIM


def test_without_the_lens_the_rim_reads_short_by_the_barrel():
    k1, k2, _, _, k3 = LENS["D"]
    for (x, y), (u, v) in zip(GRID, from_above(GRID), strict=True):
        s = ball_state(seen(u, v, 1.0), [], CFG)
        r2 = (x * x + y * y) / HEIGHT**2
        shrink = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        assert (s.x_mm, s.y_mm) == pytest.approx((x * shrink, y * shrink), abs=1e-6)
    edge = ball_state(seen(*from_above([(RIM, 0)])[0], 1.0), [], CFG)
    assert edge.x_mm < RIM - 7


def test_no_lens_is_the_plain_linear_map():
    cfg = {**CFG, "px_per_mm": 2.37, "plate_centre_px": [624, 366], "plate_yaw_deg": 30.0}
    points = [(700, 350), (712, 341), (731, 330), (1000, 100), (980, 120)]
    rim, last = track(points, cfg=cfg)[-2:]
    assert (rim.x_mm, rim.y_mm) == pytest.approx((81.27660414470422, 176.52437021378086))
    assert (last.x_mm, last.y_mm) == pytest.approx((78.18778217183976, 164.99672967551555))
    assert (last.vx_mm_s, last.vy_mm_s) == pytest.approx((1344.8007706713383, 3577.909734920051))
    flat = {**LENS, "D": [0, 0, 0, 0, 0]}
    for a, b in zip(track(points, cfg=cfg), track(points, cfg=cfg, lens=flat), strict=True):
        assert (a.x_mm, a.y_mm, a.vx_mm_s, a.vy_mm_s) == pytest.approx(
            (b.x_mm, b.y_mm, b.vx_mm_s, b.vy_mm_s), abs=1e-6
        )
