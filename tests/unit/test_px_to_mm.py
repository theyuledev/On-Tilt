import pytest

from src.contracts import BallState
from src.vision.px_to_mm import SPAN, ball_state
from src.vision.tracker import BallPx

pytestmark = pytest.mark.unit

CFG = {"px_per_mm": 2.0, "plate_centre_px": [640, 400]}


def seen(x, y, t, valid=True):
    return BallPx(x, y, 40.0, valid, t)


def track(points, dt=0.01):
    states = []
    for i, (x, y) in enumerate(points):
        states.append(ball_state(seen(x, y, i * dt), states[-SPAN:], CFG))
    return states


def test_plate_centre_is_the_origin_with_y_up_and_the_scale_in_mm():
    assert ball_state(seen(640, 400, 1.0), [], CFG) == BallState(0.0, 0.0, 0.0, 0.0, True, 1.0)
    s = ball_state(seen(660, 380, 1.0), [], CFG)
    assert (s.x_mm, s.y_mm) == (10.0, 10.0)
    s = ball_state(seen(600, 440, 1.0), [], CFG)
    assert (s.x_mm, s.y_mm) == (-20.0, -20.0)


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
