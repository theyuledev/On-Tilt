import pytest

from src.contracts import BallState
from src.control.pid import Integral, integrate, pid, smooth

pytestmark = pytest.mark.unit

C = {
    "kp_deg_mm": 0.1,
    "kd_deg_s_mm": 0.01,
    "d_max_deg": 2.0,
    "ki_deg_s_mm": 0.02,
    "i_max_deg": 1.5,
    "i_err_max_mm": 30,
}


def ball(x, y=0.0, t=0.0, valid=True):
    return BallState(x, y, 0.0, 0.0, valid=valid, t_s=t)


def test_integral_grows_with_position_times_time_from_the_second_sample():
    acc = integrate(Integral(), ball(10.0, -20.0, t=1.0), C)
    assert (acc.x, acc.y, acc.t_s) == (0.0, 0.0, 1.0)
    acc = integrate(acc, ball(10.0, -20.0, t=3.0), C)
    assert (acc.x, acc.y) == pytest.approx((0.4, -0.8))


def test_integral_only_winds_near_centre():
    acc = integrate(Integral(t_s=0.0), ball(30.0, -29.0, t=1.0), C)
    assert (acc.x, acc.y) == pytest.approx((0.0, -0.58))


def test_integral_is_clamped_and_cleared_when_the_ball_is_lost():
    acc = Integral(t_s=0.0)
    for t in range(1, 40):
        acc = integrate(acc, ball(25.0, -25.0, t=float(t)), C)
    assert (acc.x, acc.y) == (1.5, -1.5)
    assert integrate(acc, ball(0.0, valid=False), C) == Integral()


def test_pid_adds_the_integral_and_still_clamps_the_d_term():
    assert pid(10.0, 0.0, 0.5, C) == pytest.approx(-1.5)
    assert pid(0.0, 1000.0, -0.5, C) == pytest.approx(-1.5)


def test_smooth_is_a_first_order_lag_that_passes_through_at_or_under_a_tick():
    assert smooth(0.0, 30.0, 0.01, 0.03) == pytest.approx(10.0)
    assert smooth(0.0, 30.0, 0.01, 0.01) == 30.0
    assert smooth(0.0, 30.0, 0.0, 0.03) == 0.0
