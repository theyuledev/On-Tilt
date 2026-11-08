import math
import time

import pytest

from src.app import Loop
from src.io.settings import load

pytestmark = pytest.mark.sim


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def drive(loop, clock, intent, seconds):
    dt = 1 / loop.cfg["control"]["sim_tick_hz"]
    for _ in range(round(seconds / dt)):
        clock[0] += dt
        loop.keys.set(*intent)
        yield loop.tick(clock[0], dt).ball


def test_assist_parks_the_ball_at_the_corner_goal_then_brings_it_home(clock):
    cfg = load()
    cfg["control"]["mode"] = "assist"
    rig = cfg["rig"]
    edge = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2
    reach = edge - cfg["control"]["assist_margin_mm"]
    loop = Loop(cfg)
    loop.arm()
    near = False
    for ball in drive(loop, clock, (1.0, 1.0), 3.0):
        assert abs(ball.x_mm) < edge and abs(ball.y_mm) < edge
        near = near or math.dist((ball.x_mm, ball.y_mm), (reach, reach)) < 25
    assert near
    for ball in drive(loop, clock, (0.0, 0.0), 3.0):
        assert abs(ball.x_mm) < edge and abs(ball.y_mm) < edge
    assert math.hypot(ball.x_mm, ball.y_mm) < 20


@pytest.mark.parametrize("friction", [0, 45, 70])
@pytest.mark.parametrize("ball_d", [61.8, 38, 13])
def test_assist_keeps_clear_of_the_rim_with_the_friction_lean(clock, ball_d, friction):
    cfg = load()
    cfg["control"].update(mode="assist")
    cfg["control"].update(rolling_friction_x_mm_s2=friction, rolling_friction_y_mm_s2=friction)
    cfg["rig"]["ball_d_mm"] = ball_d
    edge = (cfg["rig"]["plate_inner_mm"] - ball_d) / 2
    loop = Loop(cfg)
    loop.arm()
    balls = [b for i in ((1.0, 1.0), (-1.0, 0.0), (0.0, 0.0)) for b in drive(loop, clock, i, 3.0)]
    assert min(edge - max(abs(b.x_mm), abs(b.y_mm)) for b in balls) > 15
