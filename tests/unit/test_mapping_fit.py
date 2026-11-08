import math

import numpy as np
import pytest

from src.contracts import BallState
from src.control import ROLL
from src.control.mapping import MappingWatch, fit, pushed_mm
from src.io.settings import load

T = np.linspace(0.0, 0.3, 31)
E = 400 * T**3 - 30 * T**2


@pytest.mark.parametrize("s", [1.0, -1.0, math.cos(math.radians(30))])
def test_fit_finds_the_gain_exactly(s):
    got, excite = fit(T, 3.0 - 20.0 * T + s * E, E)
    assert got == pytest.approx(s, abs=1e-9)
    assert excite > 0


def test_a_straight_line_push_says_nothing():
    assert fit(T, 5 * T, 2.0 + 7.0 * T)[1] == pytest.approx(0.0, abs=1e-12)


def test_a_steady_push_has_the_rms_of_a_parabola_off_its_line():
    a = 500.0
    excite = fit(T, 0 * T, a * T**2 / 2)[1]
    u = T / 0.3
    want = a * 0.3**2 / 2 * np.sqrt(np.mean((u**2 - u + 1 / 6) ** 2))
    assert excite == pytest.approx(want, rel=0.01)


def test_a_tilt_held_long_pushes_at_roll_g_sin():
    c = load()["control"]
    t = np.linspace(1.0, 1.3, 31)
    e = pushed_mm([(0.0, 2.0, -1.0)], t, c)
    for k, deg in enumerate((2.0, -1.0)):
        acc = np.diff(e[:, k], 2) / 0.01**2
        assert acc == pytest.approx(ROLL * c["sim_g_mm_s2"] * math.sin(math.radians(deg)), rel=1e-3)


def feed(s, amp_deg=4.0, seconds=1.0):
    """Ball on x moves s times the push of a square wave tilt; returns the trip times."""
    cfg = load()
    watch, dt = MappingWatch(cfg), 0.01
    ticks = np.arange(0.0, seconds, dt)
    cmds = [(t, amp_deg * (1 if math.sin(2 * math.pi * 2 * t) >= 0 else -1), 0.0) for t in ticks]
    x = s * pushed_mm(cmds, ticks, cfg["control"])[:, 0]
    trips = []
    for i, t in enumerate(ticks):
        if watch.step(t, BallState(x[i], 0.0, 0.0, 0.0, True, t), cmds[i][1:]):
            trips.append((round(t, 2), watch.bad))
    return trips


def test_a_flipped_axis_trips_on_the_first_window():
    assert feed(-1.0) == [(0.3, "x")]


def test_a_weak_push_needs_two_windows_in_a_row():
    assert feed(0.1) == [(0.4, "x")]


def test_the_right_push_never_trips():
    assert feed(1.0) == [] and feed(0.5) == []


def test_a_window_that_barely_tilts_is_not_judged():
    assert feed(-1.0, amp_deg=0.5) == []


def test_invalid_balls_are_skipped_and_a_gap_starts_over():
    cfg = load()
    watch = MappingWatch(cfg)
    for i in range(100):
        t = i * 0.01
        valid = not 0.2 <= t < 0.35
        watch.step(t, BallState(0.0, 0.0, 0.0, 0.0, valid, t), (0.0, 0.0))
    assert watch.since == pytest.approx(0.35)
    assert all(b[0] >= 0.35 for b in watch.balls)
