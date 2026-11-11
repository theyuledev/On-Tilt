import math

import pytest

from src.contracts import BallState
from src.control.pid import Integral, integrate, pid, smooth, unstick

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


LEAN = {
    "sim_g_mm_s2": 9810,
    "rolling_friction_x_mm_s2": 47,
    "rolling_friction_y_mm_s2": 39,
    "unstick_gain": 1.2,
    "unstick_in_mm": 1,
    "unstick_full_mm": 10,
    "unstick_v_mm_s": 80,
}
FULL = 1.2 * math.degrees(math.asin(47 / (5 / 7 * 9810)))


def test_unstick_leans_the_full_amount_toward_the_goal_when_slow_and_off_it():
    assert FULL == pytest.approx(0.461, abs=0.001)
    assert unstick(30.0, 0.0, 0.0, 0.0, LEAN) == pytest.approx((-FULL, 0.0))
    assert unstick(0.0, -20.0, 0.0, 0.0, LEAN) == pytest.approx((0.0, FULL))
    kx, ky = unstick(-20.0, 20.0, 10.0, -10.0, LEAN)
    assert kx == pytest.approx(FULL / math.sqrt(2)) and ky == pytest.approx(-FULL / math.sqrt(2))


def test_unstick_eases_in_from_the_goal_and_fades_out_with_speed():
    assert unstick(0.0, 0.0, 0.0, 0.0, LEAN) == (0.0, 0.0)
    assert unstick(1.0, 0.0, 0.0, 0.0, LEAN) == pytest.approx((0.0, 0.0))
    assert unstick(10.0, 0.0, 0.0, 0.0, LEAN) == pytest.approx((-FULL, 0.0))
    assert unstick(5.5, 0.0, 0.0, 0.0, LEAN)[0] == pytest.approx(-FULL / 2)
    assert unstick(30.0, 0.0, 40.0, 0.0, LEAN)[0] == pytest.approx(-FULL)
    assert unstick(30.0, 0.0, 60.0, 0.0, LEAN)[0] == pytest.approx(-FULL / 2)
    assert unstick(30.0, 0.0, 0.0, 80.0, LEAN) == pytest.approx((0.0, 0.0))


def test_unstick_is_bounded_and_continuous():
    lean = [math.hypot(*unstick(r / 10, 0.0, v, 0.0, LEAN)) for r in range(300) for v in (0, 50)]
    assert max(lean) <= FULL + 1e-9
    near = [abs(unstick(r / 100, 0.0, 0.0, 0.0, LEAN)[0]) for r in range(1500)]
    assert max(abs(b - a) for a, b in zip(near, near[1:], strict=False)) < 0.002


def test_unstick_is_off_at_zero_gain_or_a_zero_fade():
    assert unstick(30.0, 0.0, 0.0, 0.0, {**LEAN, "unstick_gain": 0.0}) == (0.0, 0.0)
    assert unstick(30.0, 0.0, 0.0, 0.0, {**LEAN, "unstick_v_mm_s": 0}) == pytest.approx((0.0, 0.0))
