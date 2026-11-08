import math
import random
from dataclasses import replace
from functools import cache

import pytest

from src import safety
from src.contracts import BallState, Intent
from src.control.deadband import invert, start
from src.control.pulse_map import tilt_to_pulse
from src.io.settings import load
from src.modes import make_mode
from src.sim.ball import roll
from src.sim.plate import SimActuator
from src.vision.px_to_mm import ball_state
from src.vision.tracker import BallPx

pytestmark = pytest.mark.sim

NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)
STARTS = [(80, 0, 0, 0), (0, -80, 0, 0), (-60, 60, 0, 0), (50, 50, -100, 100), (100, -100, 0, 0)]
TRIM_DEG = 1.0
NOISE_MM = 0.15
# 1:1 px to mm so the tracker's own frame difference gives the velocity
PX = {"plate_centre_px": (0.0, 0.0), "px_per_mm": 1.0, "plate_yaw_deg": 0.0}
# the longer horn: twice the pulse per degree, twice the pulse range
HORN = {
    "us_per_deg_x": 57,
    "us_per_deg_y": 57,
    "limit_x_us": 300,
    "limit_y_us": 300,
    "startup_flag": False,
}


def config(horn, scale):
    cfg = load()
    rig = {**cfg["rig"], **(HORN if horn else {})}
    c = dict(cfg["control"])
    for k in ("kp_deg_mm", "kd_deg_s_mm", "ki_deg_s_mm"):
        c[k] *= scale
    return {**cfg, "rig": rig, "control": c}


def run(cfg, start_pos, seconds=35.0):
    rig, c = cfg["rig"], cfg["control"]
    dt = 1 / c["sim_tick_hz"]
    plate, mode = SimActuator(rig, c), make_mode("balance", cfg)
    guard = safety.SafetyState()
    guard.arm()
    band, rnd = start(rig), random.Random(7)
    px = {**cfg["vision"], **PX}
    ball = BallState(*start_pos, valid=True, t_s=0.0)
    frames, balls = [], []
    for i in range(1, round(seconds / dt) + 1):
        now = i * dt
        # mode sees last tick's ball, noisy, with the tracker's velocity
        x, y = ball.x_mm + rnd.gauss(0, NOISE_MM), ball.y_mm + rnd.gauss(0, NOISE_MM)
        frames.append(ball_state(BallPx(x, -y, 0.0, True, ball.t_s), frames, px))
        tilt = mode(frames[-1], NO_INTENT, cfg)
        sent = safety.step(guard, tilt_to_pulse(tilt, rig), False, dt, rig)
        if c["inverter_on"]:
            band, sent = invert(band, sent, False, cfg)
        state = plate.send(sent, now)
        x_deg, y_deg = state.est_x_deg + TRIM_DEG, state.est_y_deg + TRIM_DEG
        ball = roll(ball, replace(state, est_x_deg=x_deg, est_y_deg=y_deg), now, rig, c)
        balls.append(ball)
    return balls


def worst(balls, after_s):
    return max(math.hypot(b.x_mm, b.y_mm) for b in balls if b.t_s > after_s)


def settle_s(balls, within_mm=20.0):
    # when the ball last came in from outside within_mm, inf if it ends outside
    out = [i for i, b in enumerate(balls) if math.hypot(b.x_mm, b.y_mm) >= within_mm]
    if not out:
        return 0.0
    return balls[out[-1] + 1].t_s if out[-1] + 1 < len(balls) else math.inf


@cache
def measure(horn, scale):
    cfg = config(horn, scale)
    holds = [run(cfg, s) for s in STARTS]
    return {
        "after_5_s": max(worst(b, 5) for b in holds),
        "after_15_s": max(worst(b, 15) for b in holds),
        "settle_60_s": settle_s(run(cfg, (60, 0, 0, 0), seconds=5.0)),
    }


CASES = [(h, s) for h in (False, True) for s in (1.0, 0.7)]


def name(case):
    return f"{'horn' if case[0] else 'bench'}-{case[1]}x"


@pytest.mark.parametrize("case", CASES, ids=name)
def test_worst_radius_after_5_s_under_15_mm(case):
    assert measure(*case)["after_5_s"] < 15


@pytest.mark.parametrize("case", CASES, ids=name)
def test_worst_radius_after_15_s_under_12_mm(case):
    assert measure(*case)["after_15_s"] < 12


@pytest.mark.parametrize("case", CASES, ids=name)
def test_60_mm_release_stays_inside_20_mm_in_time(case):
    assert measure(*case)["settle_60_s"] <= {1.0: 2.5, 0.7: 3.5}[case[1]]
