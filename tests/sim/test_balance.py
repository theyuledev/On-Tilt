import math

import pytest

from src import safety
from src.contracts import BallState, Intent
from src.control.pulse_map import tilt_to_pulse
from src.io.settings import load
from src.modes import MODES
from src.sim.ball import roll
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)


def worst_hold(start, settle_s=5.0, hold_s=30.0):
    cfg = load()
    rig, control = cfg["rig"], cfg["control"]
    dt = 1 / control["sim_tick_hz"]
    plate = SimActuator(rig, control)
    guard = safety.SafetyState()
    guard.arm()
    ball = BallState(*start, valid=True, t_s=0.0)
    worst = 0.0
    for i in range(1, round((settle_s + hold_s) / dt) + 1):
        now = i * dt
        # mode sees last tick's ball, like a camera frame
        cmd = MODES["balance"](ball, NO_INTENT, cfg)
        sent = safety.step(guard, tilt_to_pulse(cmd, rig), False, dt, rig)
        ball = roll(ball, plate.send(sent, now), now, rig, control)
        if now > settle_s:
            worst = max(worst, math.hypot(ball.x_mm, ball.y_mm))
    return worst


@pytest.mark.parametrize(
    "start",
    [(80, 0, 0, 0), (0, -80, 0, 0), (-60, 60, 0, 0), (50, 50, -100, 100), (100, -100, 0, 0)],
)
def test_balance_holds_ball_within_20_mm_for_30_s(start):
    assert worst_hold(start) < 20
