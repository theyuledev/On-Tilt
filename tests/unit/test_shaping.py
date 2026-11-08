import math
import time

import pytest

from src.contracts import BallState, Intent
from src.inputs.keyboard import KeyboardInput
from src.inputs.shaping import shape
from src.modes import MODES

pytestmark = pytest.mark.unit

CFG = {"dead_zone": 0.08, "expo": 0.3, "intent_gain_deg": 6.0, "key_ramp_s": 0.3}


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def test_dead_zone_is_radial_not_per_axis():
    assert shape(0.05, 0.05, CFG) == (0.0, 0.0)
    x, y = shape(0.1, 0.0, CFG)
    assert x > 0 and y == 0


def test_dead_zone_edge_starts_at_zero_and_keeps_the_direction():
    assert shape(0.08, 0.0, CFG) == pytest.approx((0.0, 0.0))
    x, y = shape(0.3, 0.4, {**CFG, "expo": 0.0})
    assert y / x == pytest.approx(4 / 3)


def test_shape_is_monotonic_and_odd():
    xs = [shape(v / 100, 0.0, CFG)[0] for v in range(101)]
    assert xs == sorted(xs) and xs[-1] == pytest.approx(6.0)
    assert shape(-0.5, 0.0, CFG)[0] == -shape(0.5, 0.0, CFG)[0]


def test_expo_bends_the_middle_down_but_not_the_ends():
    linear = {**CFG, "expo": 0.0}
    assert shape(0.5, 0.0, CFG)[0] < shape(0.5, 0.0, linear)[0]
    assert shape(1.0, 0.0, CFG)[0] == pytest.approx(shape(1.0, 0.0, linear)[0])


def test_gain_scales_the_output():
    assert shape(1.0, 0.0, {**CFG, "intent_gain_deg": 12.0})[0] == pytest.approx(12.0)


def test_inside_the_dead_zone_is_exactly_zero():
    for r in (0.0, 0.03, 0.079):
        assert shape(r * 0.6, r * 0.8, CFG) == (0.0, 0.0)


@pytest.mark.parametrize("edge", [0.08, 0.16])
def test_dead_zone_joins_are_smooth(edge):
    def f(r):
        return shape(r, 0.0, CFG)[0]

    h = 1e-6
    assert f(edge + h) - f(edge - h) == pytest.approx(0.0, abs=1e-4)
    left = (f(edge - h) - f(edge - 2 * h)) / h
    right = (f(edge + 2 * h) - f(edge + h)) / h
    assert left == pytest.approx(right, rel=1e-3, abs=1e-3)


def test_full_intent_is_full_gain_on_any_heading():
    assert shape(1.0, 0.0, CFG)[0] == pytest.approx(6.0)
    x, y = shape(0.6, 0.8, {**CFG, "expo": 0.0})
    assert math.hypot(x, y) == pytest.approx(6.0)


def test_held_key_ramps_up_over_key_ramp_s_and_releases_at_once(clock):
    keys = KeyboardInput(CFG)
    keys.set(1.0, 0.0)
    assert keys.latest().x == 0.0 and keys.latest().valid
    clock[0] += 0.15
    keys.set(1.0, 0.0)
    assert keys.latest().x == pytest.approx(0.5)
    clock[0] += 0.15
    keys.set(1.0, 0.0)
    assert keys.latest().x == pytest.approx(1.0)
    clock[0] += 1.0
    keys.set(1.0, 0.0)
    assert keys.latest().x == 1.0
    keys.set(0.0, 0.0)
    assert keys.latest() == Intent(0.0, 0.0, True, clock[0])


def test_each_axis_ramps_on_its_own(clock):
    keys = KeyboardInput(CFG)
    keys.set(1.0, 0.0)
    clock[0] += 0.3
    keys.set(1.0, 1.0)
    assert keys.latest().x == pytest.approx(1.0) and keys.latest().y == 0.0
    clock[0] += 0.3
    keys.set(1.0, 1.0)
    assert keys.latest().y == pytest.approx(1.0)


def test_hand_mode_shapes_key_intent_like_hand_intent(clock):
    keys = KeyboardInput(CFG)
    keys.set(0.5, -0.5)
    clock[0] += 0.3
    keys.set(0.5, -0.5)
    ball = BallState(0.0, 0.0, 0.0, 0.0, valid=False, t_s=0.0)
    cmd = MODES["hand"](ball, keys.latest(), {"input": CFG})
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx(shape(0.5, -0.5, CFG))
    assert 0 < cmd.tilt_x_deg < 3.0
