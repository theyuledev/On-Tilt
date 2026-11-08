import math
import time
from dataclasses import replace

import pytest
import test_edge_guard as eg

from src.sim.ball import roll

pytestmark = pytest.mark.sim

# rolling friction of the real balls, mm/s^2; the sim plate has none
ROUGH = (47.0, 39.0)


def rough_roll(ball, plate, now, rig, control):
    dt = now - ball.t_s
    b = roll(ball, plate, now, rig, control)
    v = [
        0.0 if abs(u) <= f * dt else u - math.copysign(f * dt, u)
        for u, f in zip((b.vx_mm_s, b.vy_mm_s), ROUGH, strict=True)
    ]
    return replace(b, vx_mm_s=v[0], vy_mm_s=v[1])


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


@pytest.fixture(params=[False, True], ids=["smooth", "rough"])
def rough(request, monkeypatch):
    if request.param:
        monkeypatch.setattr(eg, "roll", rough_roll)
    return request.param


def guarded(ball_d):
    cfg = eg.rig_cfg(ball_d, cap=2.0)
    cfg["control"]["guard_on"] = True
    return cfg


def slow_sway(pulses, k=50):
    avg = [sum(pulses[i : i + k]) / k for i in range(len(pulses) - k + 1)]
    return max(avg) - min(avg)


@pytest.mark.parametrize("ball_d", eg.BALLS)
def test_a_pusher_held_at_the_line_does_not_sway(clock, rough, ball_d):
    # the guard alone swayed 150 us smooth / 80 us rough and held 10-13 mm off the rim
    cfg = guarded(ball_d)
    r = eg.run(cfg, clock, lambda t, b: (1.0, 0.0), seconds=12.0)
    pulses = r["pulses"][300:]
    assert slow_sway(pulses) < (40 if rough else 90)
    assert r["gap"] > (15 if rough else 12)
    assert eg.reversals(pulses, cfg["control"]["servo_dead_band_us"]) / 9.0 < 2


@pytest.mark.parametrize("ball_d", eg.BALLS)
def test_the_ring_and_a_hard_player_on_rough_balls(clock, rough, ball_d):
    cfg = eg.linear_hand(guarded(ball_d))
    hold = eg.pd((100.0, 100.0))
    r = eg.run(cfg, clock, lambda t, b: hold(b), seconds=4.0, start=(100.0, 100.0))
    assert max(math.dist(p, (100, 100)) for p in r["track"][50:]) < 5
    player, _ = eg.lap(ahead=80.0, kp=0.1, kd=0.0375)
    assert eg.run(cfg, clock, player, seconds=10.0, start=(100.0, 100.0))["gap"] > 5
