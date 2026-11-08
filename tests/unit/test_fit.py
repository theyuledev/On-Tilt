import math

import numpy as np
import pytest

from src.contracts import BallState, PlateCommand, PlateState
from src.control.pulse_map import tilt_to_pulse
from src.io.settings import load
from src.sim.ball import roll
from src.sim.fit import ball_accel, settle, step_response
from src.sim.plate import SimActuator

pytestmark = pytest.mark.unit


def test_step_response_recovers_the_sim_servo_numbers():
    cfg = load()
    rig, control = cfg["rig"], cfg["control"]
    dt = 1 / control["sim_tick_hz"]
    act = SimActuator(rig, control)
    hold, step = (tilt_to_pulse(PlateCommand(deg, 0.0), rig) for deg in (0.0, 3.0))
    t = [i * dt for i in range(150)]
    states = [act.send(hold if i < 30 else step, now) for i, now in enumerate(t)]

    fit = step_response(t, [s.sent_x_deg for s in states], [s.est_x_deg for s in states])

    # one tick: the sim retargets on whole ticks, so its motion starts on one
    assert fit.dead_s == pytest.approx(control["servo_dead_time_s"], abs=dt)
    # the rate cap holds the first ticks of a 3 deg step and delays the 63 pct point
    assert fit.tau_s == pytest.approx(control["servo_tau_s"], rel=0.15)
    top = control["servo_top_speed_us_s"] / rig["us_per_deg_x"]
    assert fit.top_speed_deg_s == pytest.approx(top, rel=0.1)


def test_step_response_that_never_moves_has_no_times():
    t = np.arange(0, 1, 0.01)
    fit = step_response(t, np.where(t < 0.3, 0.0, 3.0), np.zeros_like(t))
    assert fit.dead_s == math.inf and fit.tau_s == math.inf and fit.top_speed_deg_s == 0


def test_ball_accel_matches_a_rolling_ball_under_one_degree():
    cfg = load()
    rig, control = cfg["rig"], cfg["control"]
    dt = 1 / control["sim_tick_hz"]
    plate = PlateState(1.0, 0.0, 1.0, 0.0)
    ball = BallState(0.0, 0.0, 0.0, 0.0, valid=True, t_s=0.0)
    t, x = [0.0], [0.0]
    for i in range(1, 101):
        ball = roll(ball, plate, i * dt, rig, control)
        t.append(ball.t_s)
        x.append(ball.x_mm)

    fit = ball_accel(t, x, 1.0)

    expect = 5 / 7 * control["sim_g_mm_s2"] * math.sin(math.radians(1.0))
    assert fit.a_per_deg == pytest.approx(expect, rel=0.03)
    # time to roll the 0.3 mm motion floor from rest
    assert fit.first_motion_s == pytest.approx(math.sqrt(2 * 0.3 / expect), abs=dt)


def test_ball_accel_with_a_still_ball_is_zero_and_never_moves():
    t = np.arange(0, 1, 0.01)
    fit = ball_accel(t, np.zeros_like(t), 1.0)
    assert fit.a_per_deg == 0 and fit.first_motion_s == math.inf


def test_settle_reads_a_damped_oscillation():
    t = np.arange(0, 10, 0.01)
    x = 60 * np.exp(-t) * np.cos(2 * np.pi * t / 1.2)
    r = np.abs(x)

    s = settle(t, x, np.zeros_like(t))

    for got, lim in ((s.t20_s, 20), (s.t5_s, 5)):
        k = np.searchsorted(t, got)
        assert r[k - 1] >= lim and r[k:].max() < lim
    # first swing past centre: e^-t cos(wt) bottoms out where tan(wt) = -1/w
    w = 2 * np.pi / 1.2
    tm = (np.pi - np.arctan(1 / w)) / w
    assert s.overshoot_pct == pytest.approx(-100 * np.exp(-tm) * np.cos(w * tm), abs=0.1)
    assert s.period_s == pytest.approx(1.2, abs=0.02)
    assert 0 < s.rms_mm < 0.5


def test_settle_that_never_gets_inside_gives_inf():
    t = np.arange(0, 6, 0.01)
    s = settle(t, np.full_like(t, 30.0), np.zeros_like(t))
    assert s.t20_s == math.inf and s.t5_s == math.inf and s.period_s == math.inf
    assert s.overshoot_pct == 0
    assert s.rms_mm == pytest.approx(30.0)


def test_settle_from_the_centre_has_no_swing_numbers():
    t = np.arange(0, 4, 0.01)
    s = settle(t, 10 * t, np.zeros_like(t))
    assert s.overshoot_pct == 0 and s.period_s == math.inf
    assert s.t20_s == math.inf and s.t5_s == math.inf
    assert s.rms_mm == pytest.approx(math.sqrt(np.mean((10 * t) ** 2)))


def test_settle_already_inside_the_band_is_zero_seconds():
    t = np.arange(0, 6, 0.01)
    s = settle(t, np.full_like(t, 3.0), np.zeros_like(t))
    assert s.t20_s == 0 and s.t5_s == 0
    assert s.rms_mm == pytest.approx(3.0)
