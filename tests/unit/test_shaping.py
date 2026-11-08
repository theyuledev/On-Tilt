import math
import time

import pytest

from src.contracts import BallState, Intent
from src.inputs.keyboard import KeyboardInput
from src.inputs.shaping import floor_deg, shape, top_deg
from src.modes import MODES

pytestmark = pytest.mark.unit

INPUT = {"dead_zone": 0.08, "expo": 0.3, "key_ramp_s": 0.3}
RIG = {
    "center_x_us": 1500,
    "center_y_us": 1500,
    "trim_x_us": 0,
    "trim_y_us": 0,
    "startup_flag": False,
    "startup_range_us": 150,
    "limit_x_us": 150,
    "limit_y_us": 150,
}
WIDE = {**RIG, "us_per_deg_x": 25.0, "us_per_deg_y": 25.0}  # 6 deg at the clamp
REAL = {**RIG, "us_per_deg_x": 100.0, "us_per_deg_y": -100.0}
SLICK = {"sim_g_mm_s2": 9810, "rolling_friction_x_mm_s2": 0.0, "rolling_friction_y_mm_s2": 0.0}
ROUGH = {"sim_g_mm_s2": 9810, "rolling_friction_x_mm_s2": 47, "rolling_friction_y_mm_s2": 39}


def cfg(rig=WIDE, control=SLICK, **inp):
    return {"input": {**INPUT, **inp}, "rig": rig, "control": control}


CFG = cfg()
FELT = cfg(REAL, ROUGH)
DZ = INPUT["dead_zone"]


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
    x, y = shape(0.3, 0.4, cfg(expo=0.0))
    assert y / x == pytest.approx(4 / 3)


def test_shape_is_monotonic_and_odd():
    xs = [shape(v / 100, 0.0, CFG)[0] for v in range(101)]
    assert xs == sorted(xs) and xs[-1] == pytest.approx(6.0)
    assert shape(-0.5, 0.0, CFG)[0] == -shape(0.5, 0.0, CFG)[0]


def test_expo_bends_the_middle_down_but_not_the_ends():
    linear = cfg(expo=0.0)
    assert shape(0.5, 0.0, CFG)[0] < shape(0.5, 0.0, linear)[0]
    assert shape(1.0, 0.0, CFG)[0] == pytest.approx(shape(1.0, 0.0, linear)[0])


def test_the_clamp_scales_the_output():
    assert shape(1.0, 0.0, cfg({**WIDE, "us_per_deg_x": 12.5}))[0] == pytest.approx(12.0)


def test_inside_the_dead_zone_is_exactly_zero():
    for c in (CFG, FELT):
        for r in (0.0, 0.03, 0.079):
            assert shape(r * 0.6, r * 0.8, c) == (0.0, 0.0)


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
    x, y = shape(0.6, 0.8, cfg(expo=0.0))
    assert math.hypot(x, y) == pytest.approx(6.0)


def test_floor_is_the_tilt_that_just_beats_rolling_friction():
    assert floor_deg(FELT) == pytest.approx((0.384, 0.319), abs=1e-3)
    assert floor_deg(CFG) == (0.0, 0.0)


def test_with_the_guard_off_top_is_the_servo_clamp():
    assert top_deg(CFG) == (6.0, 6.0)
    assert top_deg(FELT) == (1.5, 1.5)
    assert top_deg(cfg(REAL, {"player_max_deg": 1.0})) == (1.5, 1.5)
    assert top_deg(cfg(REAL, {"guard_on": False, "player_max_deg": 1.0})) == (1.5, 1.5)


def test_with_the_guard_on_top_is_the_player_cap_or_the_clamp_whichever_is_less():
    guarded = {"guard_on": True, "player_max_deg": 1.0}
    assert top_deg(cfg(REAL, guarded)) == (1.0, 1.0)
    assert top_deg(cfg(WIDE, {**guarded, "player_max_deg": 4.0})) == (4.0, 4.0)
    assert top_deg(cfg(REAL, {**guarded, "player_max_deg": 4.0})) == (1.5, 1.5)


def test_top_follows_the_startup_range_while_the_startup_flag_is_on():
    wide = {**REAL, "limit_x_us": 300, "limit_y_us": 300}
    assert top_deg(cfg({**wide, "startup_flag": True})) == (1.5, 1.5)
    assert top_deg(cfg({**wide, "startup_flag": False})) == (3.0, 3.0)


def floor_part(x, y):
    # what friction adds on top of the plain curve: out = floor + (top - floor) * curve
    (sx, sy), (ox, oy) = shape(x, y, cfg(REAL)), shape(x, y, FELT)
    fx, fy = floor_deg(FELT)
    return ox - (1 - fx / 1.5) * sx, oy - (1 - fy / 1.5) * sy


def test_no_dead_zone_does_not_divide_by_zero():
    for c in (cfg(dead_zone=0.0), cfg(REAL, ROUGH, dead_zone=0.0)):
        assert shape(0.0, 0.0, c) == (0.0, 0.0)
        x, y = shape(0.5, 0.0, c)
        assert math.isfinite(x) and x > 0 and y == 0


def test_odd_settings_never_divide_by_zero():
    assert shape(1.0, 0.0, cfg(dead_zone=0.7)) == pytest.approx((6.0, 0.0))
    assert shape(1.0, 1.0, cfg(dead_zone=1.0)) == (0.0, 0.0)
    assert top_deg(cfg({**REAL, "us_per_deg_x": 0.0, "us_per_deg_y": 0.0})) == (0.0, 0.0)
    assert floor_deg(cfg(control={**ROUGH, "sim_g_mm_s2": 0})) == (0.0, 0.0)
    assert floor_deg(cfg(control={**ROUGH, "sim_g_mm_s2": 1})) == (90.0, 90.0)


def test_no_step_at_the_dead_zone_edge_with_friction():
    h = 1e-6
    for x, y in ((DZ, 0.0), (0.0, -DZ)):
        lo, hi = shape(x * (1 - h), y * (1 - h), FELT), shape(x * (1 + h), y * (1 + h), FELT)
        assert hi == pytest.approx(lo, abs=1e-4)


@pytest.mark.parametrize("r", [0.1, 0.12, 0.16, 0.3, 1.0])
def test_the_floor_fades_in_over_one_dead_zone_width_then_stays_full(r):
    fx, fy = floor_deg(FELT)
    b = min(1.0, (r - DZ) / DZ)
    assert floor_part(r, 0.0) == pytest.approx((fx * b, 0.0))
    assert floor_part(0.0, -r) == pytest.approx((0.0, -fy * b))


def test_a_diagonal_splits_the_floor_instead_of_doubling_it():
    fx, fy = floor_deg(FELT)
    assert floor_part(0.7, 0.7) == pytest.approx((fx * math.sqrt(0.5), fy * math.sqrt(0.5)))


def test_full_stick_reaches_the_top_on_each_axis():
    assert shape(1.0, 0.0, FELT) == pytest.approx((1.5, 0.0))
    assert shape(0.0, -1.0, FELT) == pytest.approx((0.0, -1.5))
    assert shape(1.0, 0.0, cfg(REAL))[0] == pytest.approx(1.5)


def test_the_whole_hand_travel_moves_the_plate():
    clamp = REAL["limit_x_us"] / REAL["us_per_deg_x"]
    new = [shape(i, 0.0, FELT)[0] for i in (0.4, 0.6, 0.9, 1.0)]
    assert new[0] < new[1] < new[2] < clamp and new[3] == pytest.approx(clamp)
    # gain 6 with no floor is pinned at the plate clamp from about 0.42 on
    old = [min(clamp, shape(i, 0.0, CFG)[0]) for i in (0.5, 0.6, 0.9)]
    assert old == [clamp] * 3


def test_with_friction_the_curve_still_only_climbs():
    xs = [shape(v / 1000, 0.0, FELT)[0] for v in range(1001)]
    assert xs == sorted(xs) and xs[0] == 0.0


def test_held_key_ramps_up_over_key_ramp_s_and_releases_at_once(clock):
    keys = KeyboardInput(INPUT)
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
    keys = KeyboardInput(INPUT)
    keys.set(1.0, 0.0)
    clock[0] += 0.3
    keys.set(1.0, 1.0)
    assert keys.latest().x == pytest.approx(1.0) and keys.latest().y == 0.0
    clock[0] += 0.3
    keys.set(1.0, 1.0)
    assert keys.latest().y == pytest.approx(1.0)


def test_hand_mode_shapes_key_intent_like_hand_intent(clock):
    keys = KeyboardInput(INPUT)
    keys.set(0.5, -0.5)
    clock[0] += 0.3
    keys.set(0.5, -0.5)
    ball = BallState(0.0, 0.0, 0.0, 0.0, valid=False, t_s=0.0)
    cmd = MODES["hand"](ball, keys.latest(), FELT)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx(shape(0.5, -0.5, FELT))
    assert 0 < cmd.tilt_x_deg < 1.5
