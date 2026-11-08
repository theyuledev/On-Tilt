import math
import random

import pytest

from src.contracts import PlateCommand
from src.estimate import BallEstimator
from src.sim.ball import ROLL

pytestmark = pytest.mark.unit

CFG = {
    "control": {
        "servo_dead_time_s": 0.02,
        "servo_tau_s": 0.035,
        "sim_g_mm_s2": 9810,
        "est_accel_noise_mm_s2": 150,
        "est_pos_noise_mm": 0.15,
        "est_cam_latency_s": 0.015,
        "est_gate_mm": 30,
        "est_hold_s": 0.2,
        "est_v_max_mm_s": 1500,
    },
    "vision": {"lost_after_misses": 5},
}
C = CFG["control"]
FPS = 120
FINE = 0.001


def tilted(deg, seconds):
    # exact ball under a step tilt sent at t=0 through the same servo model, 1 ms steps
    x = v = theta = 0.0
    xs, vs = [0.0], [0.0]
    for i in range(1, round(seconds / FINE) + 1):
        if i * FINE > C["servo_dead_time_s"]:
            theta += (deg - theta) * (1 - math.exp(-FINE / C["servo_tau_s"]))
        v += ROLL * C["sim_g_mm_s2"] * math.sin(math.radians(theta)) * FINE
        x += v * FINE
        xs.append(x)
        vs.append(v)
    return xs, vs


def at(series, t):
    return series[round(t / FINE)]


def frames(seconds, gap=None):
    for i in range(round(seconds * FPS)):
        t = i / FPS
        if gap is None or not gap[0] <= t < gap[1]:
            yield t


def noisy_rest(est, seconds, seed=0):
    rng = random.Random(seed)
    for t in frames(seconds):
        yield t, est.update(t, rng.gauss(0, 0.15), rng.gauss(0, 0.15))


def test_rest_with_pixel_noise_gives_still_ball():
    est = BallEstimator(CFG)
    v2, worst = [], 0.0
    for t, s in noisy_rest(est, 2.0):
        if t >= 1.0:
            v2.append(s.vx_mm_s**2 + s.vy_mm_s**2)
            worst = max(worst, abs(s.x_mm), abs(s.y_mm))
    assert math.sqrt(sum(v2) / len(v2)) < 3
    assert worst < 0.3


def test_constant_tilt_tracks_velocity_without_bias_and_predicts_the_plate_lag():
    xs, vs = tilted(2.0, 0.7)
    est = BallEstimator(CFG)
    est.command(0.0, PlateCommand(2.0, 0.0))
    lat = C["est_cam_latency_s"]
    for t in frames(0.6):
        # the frame stamped at t shows the ball as it was one camera latency earlier
        s = est.update(t, at(xs, max(t - lat, 0)), 0.0)
        if t > 0.3:
            ahead = t + C["servo_dead_time_s"]
            assert abs(s.vx_mm_s - at(vs, ahead)) < 2
            assert abs(s.x_mm - at(xs, ahead)) < 1
            assert s.y_mm == pytest.approx(0, abs=0.01) and s.t_s == ahead


def test_74_ms_gap_keeps_the_velocity_on_track():
    xs, vs = tilted(2.0, 0.7)
    est = BallEstimator(CFG)
    est.command(0.0, PlateCommand(2.0, 0.0))
    for t in frames(0.6, gap=(0.3, 0.374)):
        s = est.update(t, at(xs, max(t - C["est_cam_latency_s"], 0)), 0.0)
        if t > 0.3:
            assert abs(s.vx_mm_s - at(vs, s.t_s)) < 15


def test_long_gap_goes_invalid_then_relocks_in_three_frames():
    est = BallEstimator(CFG)
    for _ in noisy_rest(est, 1.0):
        pass
    assert est.predict(1.0 + 0.19).valid
    assert not est.predict(1.0 + 0.25).valid
    for i in range(3):
        s = est.update(1.25 + i / FPS, 20.0, -20.0)
    assert s.valid and abs(s.x_mm - 20) < 0.5 and abs(s.y_mm + 20) < 0.5


def test_one_wild_detection_is_skipped_two_relock():
    est = BallEstimator(CFG)
    *_, (t, _s) = noisy_rest(est, 1.0)
    skipped = est.update(t + 1 / FPS, 35.0, 0.0)
    assert abs(skipped.x_mm) < 0.5
    relocked = est.update(t + 2 / FPS, 35.0, 0.0)
    assert relocked.x_mm == pytest.approx(35.0) and relocked.vx_mm_s == 0


def test_velocity_is_clamped_after_a_jump():
    est = BallEstimator(CFG)
    est.update(0.0, 0.0, 0.0)
    est.update(1 / FPS, 0.0, 0.0)
    s = est.update(2 / FPS, 29.0, -29.0)
    assert abs(s.vx_mm_s) == C["est_v_max_mm_s"] and abs(s.vy_mm_s) == C["est_v_max_mm_s"]


def test_lost_lock_holds_then_goes_invalid():
    est = BallEstimator(CFG)
    *_, (t, last) = noisy_rest(est, 1.0)
    for i in range(1, CFG["vision"]["lost_after_misses"]):
        held = est.miss(t + i / FPS)
        assert held.valid and abs(held.x_mm - last.x_mm) < 0.5
    gone = est.miss(t + 5 / FPS)
    assert not gone.valid and gone.t_s == t + 5 / FPS
