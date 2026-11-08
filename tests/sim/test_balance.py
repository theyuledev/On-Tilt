import math

import pytest

from src import safety
from src.contracts import BallState, Intent, PlateCommand
from src.control.margin import phase_margin_deg
from src.control.pulse_map import tilt_to_pulse
from src.estimate import BallEstimator
from src.io.settings import load
from src.modes import MODES
from src.sim.ball import roll
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)


def run(start, seconds, estimate=False):
    """Yields (what the mode acted on, where the ball really is) per tick."""
    cfg = load()
    rig, control = cfg["rig"], cfg["control"]
    dt = 1 / control["sim_tick_hz"]
    # sim frames are stamped exactly and the sim servo's dead time already holds the camera
    cfg["vision"] = {**cfg["vision"], "camera_latency_s": 0.0}
    est = BallEstimator(cfg)
    plate = SimActuator(rig, control)
    guard = safety.SafetyState()
    guard.arm()
    ball = BallState(*start, valid=True, t_s=0.0)
    for i in range(1, round(seconds / dt) + 1):
        now = i * dt
        # mode sees last tick's ball; the camera's own latency sits in the servo dead time
        obs = ball
        if estimate:
            est.update(obs.t_s, obs.x_mm, obs.y_mm)
            obs = est.predict(now)
        cmd = MODES["balance"](obs, NO_INTENT, cfg)
        sent = safety.step(guard, tilt_to_pulse(cmd, rig), False, dt, rig)
        state = plate.send(sent, now)
        est.command(now, PlateCommand(state.sent_x_deg, state.sent_y_deg))
        ball = roll(ball, state, now, rig, control)
        yield obs, ball


def worst_hold(start, settle_s=5.0, hold_s=30.0, estimate=False):
    return max(
        math.hypot(b.x_mm, b.y_mm)
        for _, b in run(start, settle_s + hold_s, estimate)
        if b.t_s > settle_s
    )


def settle(start, estimate, seconds=5.0, within_mm=20.0):
    """When the ball starts staying within within_mm, and how far what the mode acted on was
    from where the ball really was once the plate moved (a servo dead time later)."""
    cfg = load()
    wall = (cfg["rig"]["plate_inner_mm"] - cfg["rig"]["ball_d_mm"]) / 2
    dt = 1 / cfg["control"]["sim_tick_hz"]
    pairs = list(run(start, seconds, estimate))
    balls = [b for _, b in pairs]
    settled, seen_err = None, 0.0
    for i, (obs, b) in enumerate(pairs):
        r = math.hypot(b.x_mm, b.y_mm)
        assert r < wall
        if r >= within_mm:
            settled = None
        elif settled is None:
            settled = b.t_s
        later = i + round(cfg["control"]["servo_dead_time_s"] / dt)
        if later < len(balls):
            true = balls[later]
            seen_err += math.hypot(obs.x_mm - true.x_mm, obs.y_mm - true.y_mm)
    return (math.inf if settled is None else settled), seen_err


@pytest.mark.parametrize(
    "start",
    [
        (60, 0, 0, 0),
        (80, 0, 0, 0),
        (0, -80, 0, 0),
        (-60, 60, 0, 0),
        (50, 50, -100, 100),
        (100, -100, 0, 0),
    ],
)
def test_balance_holds_ball_within_20_mm_for_30_s(start):
    assert worst_hold(start) < 20
    assert worst_hold(start, estimate=True) < 20


def test_loop_phase_margin_is_at_least_45_deg():
    c = load()["control"]
    # dead time is command to first visible motion, so the camera is already in it
    delay = c["servo_dead_time_s"] + 1 / c["sim_tick_hz"]
    margin = phase_margin_deg(
        c["kp_deg_mm"], c["kd_deg_s_mm"], delay, c["servo_tau_s"], c["sim_g_mm_s2"]
    )
    assert margin >= 45


@pytest.mark.parametrize("start", [(60, 0, 0, 0), (-40, 45, 0, 0)])
def test_predicted_state_is_far_closer_to_the_real_ball_than_the_late_frame(start):
    with_est, err_est = settle(start, estimate=True)
    without, err_raw = settle(start, estimate=False)
    print(f"settle {start}: with estimator {with_est:.2f} s, without {without:.2f} s")
    print(f"seen error {start}: predicted {err_est:.0f} mm, late frame {err_raw:.0f} mm")
    assert with_est <= without * 1.2
    assert err_est <= err_raw / 1.4
