import time
from dataclasses import replace

import pytest
import test_edge_guard as eg

from src.control.guard import Guard
from src.sim.ball import roll

pytestmark = pytest.mark.sim


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def mirror(b):
    # the loop sees x the wrong way round
    return replace(b, x_mm=-b.x_mm, vx_mm_s=-b.vx_mm_s)


def time_to_rim(clock, monkeypatch, guard_on, seconds=6.0):
    monkeypatch.setattr(eg, "roll", lambda b, *a: mirror(roll(mirror(b), *a)))
    if not guard_on:
        # off, but the player cap stays so the two runs are comparable
        monkeypatch.setattr(Guard, "_axis", lambda self, i, p, v, player, dt: player)
    cfg = eg.linear_hand(eg.rig_cfg(cap=2.0))
    cfg["control"]["guard_on"] = True
    hold = eg.pd((105.0, 0.0))
    # the player sees the real plate and holds the ball short of the rim
    r = eg.run(cfg, clock, lambda t, b: hold(mirror(b)), seconds=seconds, start=(-60.0, 0.0))
    wall = (cfg["rig"]["plate_inner_mm"] - cfg["rig"]["ball_d_mm"]) / 2
    dt = 1 / cfg["control"]["sim_tick_hz"]
    hit = [k for k, (x, y) in enumerate(r["track"]) if max(abs(x), abs(y)) >= wall - 0.5]
    return hit[0] * dt if hit else float("inf")


@pytest.mark.xfail(
    strict=True,
    reason="the guard trusts the stored mapping; needs autocal's yaw/mirror verdict and the "
    "mapping watch's stand-down",
)
def test_a_mirrored_x_with_the_guard_on_is_no_worse_than_off(clock, monkeypatch):
    # guard on brakes the wrong way and hits the rim at 1.08 s, off holds short for all 6 s
    assert time_to_rim(clock, monkeypatch, True) >= time_to_rim(clock, monkeypatch, False)
