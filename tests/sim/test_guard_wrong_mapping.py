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


def swap(b):
    # y mirrored and turned 90 deg: what the loop sees has x and y traded
    return replace(b, x_mm=b.y_mm, y_mm=b.x_mm, vx_mm_s=b.vy_mm_s, vy_mm_s=b.vx_mm_s)


def gap(clock, monkeypatch, guard_on):
    monkeypatch.setattr(eg, "roll", lambda b, *a: swap(roll(swap(b), *a)))
    if not guard_on:
        monkeypatch.setattr(Guard, "__call__", lambda self, ball, cmd: cmd)
    return eg.run(eg.rig_cfg(), clock, lambda t, b: (1.0, 0.3), seconds=6.0)["gap"]


def test_a_wrong_mapping_with_the_guard_on_is_no_worse_than_off(clock, monkeypatch):
    # both reach the rim (gap 0.0 on and off): the guard trusts the stored mapping
    assert gap(clock, monkeypatch, True) >= gap(clock, monkeypatch, False)
