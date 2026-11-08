import math
import random
from dataclasses import replace

import pytest

from src import safety
from src.contracts import BallState, Intent, PlateCommand, ServoCommand
from src.control.deadband import invert, start
from src.control.margin import phase_margin_deg
from src.control.pulse_map import center, tilt_to_pulse
from src.estimate import BallEstimator
from src.io.settings import load
from src.modes import make_mode
from src.sim.ball import roll
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)
STARTS = [(80, 0, 0, 0), (0, -80, 0, 0), (-60, 60, 0, 0), (50, 50, -100, 100), (100, -100, 0, 0)]


def config(integral=True):
    cfg = load()
    control = dict(cfg["control"])
    if not integral:
        control["ki_deg_s_mm"] = 0.0
    return dict(cfg, control=control)


def run(start_pos, inverter=True, integral=True, noise_mm=0.0, offset_deg=0.0, hold_s=0.0):
    """Ball states per tick plus how often the x servo pulse reversed. Servo numbers and
    gains come from settings. offset_deg is a wrong trim on both axes, hold_s pins the
    ball where it started (a hand holding it) before letting go."""
    cfg = config(integral)
    rig, control = cfg["rig"], cfg["control"]
    dt = 1 / control["sim_tick_hz"]
    plate = SimActuator(rig, control)
    guard = safety.SafetyState()
    guard.arm()
    mode = make_mode("balance", cfg)
    band = start(rig)
    rnd = random.Random(7)
    ball = BallState(*start_pos, valid=True, t_s=0.0)
    balls, reversals, heading, last = [], 0, 0, None
    for i in range(1, round(35.0 / dt) + 1):
        now = i * dt
        # mode sees last tick's ball, like a camera frame
        seen = ball
        if noise_mm:
            jitter = rnd.gauss(0, noise_mm), rnd.gauss(0, noise_mm)
            seen = replace(ball, x_mm=ball.x_mm + jitter[0], y_mm=ball.y_mm + jitter[1])
        sent = safety.step(guard, tilt_to_pulse(mode(seen, NO_INTENT, cfg), rig), False, dt, rig)
        if inverter:
            band, sent = invert(band, sent, False, cfg)
        if last is not None:
            step = (sent.x_us > last) - (sent.x_us < last)
            reversals += bool(step and heading and step != heading)
            heading = step or heading
        last = sent.x_us
        state = plate.send(sent, now)
        if offset_deg:
            x, y = state.est_x_deg + offset_deg, state.est_y_deg + offset_deg
            state = replace(state, est_x_deg=x, est_y_deg=y)
        ball = replace(ball, t_s=now) if now <= hold_s else roll(ball, state, now, rig, control)
        balls.append(ball)
    return balls, reversals


def worst(balls, after_s, dt=0.01):
    return max(math.hypot(b.x_mm, b.y_mm) for b in balls[round(after_s / dt) :])


@pytest.mark.parametrize("start_pos", STARTS)
def test_holds_within_20_mm_for_30_s_even_with_a_1_deg_trim_error(start_pos):
    assert worst(run(start_pos, offset_deg=1.0)[0], 5) < 20


def test_pd_alone_cannot_hold_through_the_trim_error():
    assert worst(run(STARTS[0], inverter=False, integral=False, offset_deg=1.0)[0], 5) >= 20


def test_release_after_holding_100_mm_out_overshoots_centre_by_under_20_mm():
    balls, _ = run((100, 0, 0, 0), hold_s=5.0)
    assert min(b.x_mm for b in balls) > -20


def test_inverter_tightens_the_hold_without_buzzing_the_servo():
    with_inv = [run(s) for s in STARTS]
    without = [run(s, inverter=False) for s in STARTS]
    plain = [run(s, inverter=False, integral=False) for s in STARTS]
    assert max(worst(b, 5) for b, _ in with_inv) < max(worst(b, 5) for b, _ in without)
    assert sum(r for _, r in with_inv) < 3 * sum(r for _, r in plain)


def test_position_noise_does_not_break_the_hold():
    noisy = [run(s, noise_mm=0.15) for s in STARTS]
    assert max(worst(b, 5) for b, _ in noisy) < 20


def test_kill_ramp_is_untouched_by_the_inverter_and_ends_level():
    cfg = load()
    rig = cfg["rig"]
    dt = 1 / cfg["control"]["sim_tick_hz"]
    guard = safety.SafetyState()
    guard.arm()
    band = start(rig)
    target = ServoCommand(1600, 1420)
    for _ in range(100):
        band, sent = invert(band, safety.step(guard, target, False, dt, rig), False, cfg)
    guard.kill()
    last = sent
    while guard.armed:
        sent = safety.step(guard, target, False, dt, rig)
        band, out = invert(band, sent, guard.killing, cfg)
        assert out == sent
        last = sent or last
    assert (last.x_us, last.y_us) == center(rig)


def run_seen(start_pos, seconds, estimate=False):
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
    mode = make_mode("balance", cfg)
    ball = BallState(*start_pos, valid=True, t_s=0.0)
    for i in range(1, round(seconds / dt) + 1):
        now = i * dt
        # mode sees last tick's ball; the camera's own latency sits in the servo dead time
        obs = ball
        if estimate:
            est.update(obs.t_s, obs.x_mm, obs.y_mm)
            obs = est.predict(now)
        cmd = mode(obs, NO_INTENT, cfg)
        sent = safety.step(guard, tilt_to_pulse(cmd, rig), False, dt, rig)
        state = plate.send(sent, now)
        est.command(now, PlateCommand(state.sent_x_deg, state.sent_y_deg))
        ball = roll(ball, state, now, rig, control)
        yield obs, ball


def worst_hold(start_pos, settle_s=5.0, hold_s=30.0, estimate=False):
    return max(
        math.hypot(b.x_mm, b.y_mm)
        for _, b in run_seen(start_pos, settle_s + hold_s, estimate)
        if b.t_s > settle_s
    )


def settle(start_pos, estimate, seconds=5.0, within_mm=20.0):
    """When the ball starts staying within within_mm, and how far what the mode acted on was
    from where the ball really was once the plate moved (a servo dead time later)."""
    cfg = load()
    wall = (cfg["rig"]["plate_inner_mm"] - cfg["rig"]["ball_d_mm"]) / 2
    dt = 1 / cfg["control"]["sim_tick_hz"]
    pairs = list(run_seen(start_pos, seconds, estimate))
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
    "start_pos",
    [
        (60, 0, 0, 0),
        (80, 0, 0, 0),
        (0, -80, 0, 0),
        (-60, 60, 0, 0),
        (50, 50, -100, 100),
        (100, -100, 0, 0),
    ],
)
def test_balance_holds_ball_within_20_mm_for_30_s(start_pos):
    assert worst_hold(start_pos) < 20
    assert worst_hold(start_pos, estimate=True) < 20


def test_loop_phase_margin_is_at_least_45_deg():
    c = load()["control"]
    # dead time is command to first visible motion, so the camera is already in it
    delay = c["servo_dead_time_s"] + 1 / c["sim_tick_hz"]
    margin = phase_margin_deg(
        c["kp_deg_mm"], c["kd_deg_s_mm"], delay, c["servo_tau_s"], c["sim_g_mm_s2"]
    )
    assert margin >= 45


@pytest.mark.parametrize("start_pos", [(60, 0, 0, 0), (-40, 45, 0, 0)])
def test_predicted_state_is_far_closer_to_the_real_ball_than_the_late_frame(start_pos):
    with_est, err_est = settle(start_pos, estimate=True)
    without, err_raw = settle(start_pos, estimate=False)
    print(f"settle {start_pos}: with estimator {with_est:.2f} s, without {without:.2f} s")
    print(f"seen error {start_pos}: predicted {err_est:.0f} mm, late frame {err_raw:.0f} mm")
    assert with_est <= without * 1.2
    assert err_est <= err_raw / 1.4
