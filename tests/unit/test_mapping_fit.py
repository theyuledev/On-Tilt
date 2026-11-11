import math

import numpy as np
import pytest

from src.contracts import BallState
from src.control import ROLL, mapping
from src.control.mapping import MappingWatch, angle_deg, pushed_mm, responses
from src.io.settings import load

T = np.linspace(0.0, 0.3, 31)
E = np.column_stack([400 * T**3 - 30 * T**2, -200 * T**3 + 60 * T**2])


def rot(deg, gain=1.0):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return gain * np.array([[c, -s], [s, c]])


def angles(S, e=E, excite=1e-9):
    xy = np.array([3.0, -1.0]) + np.outer(T, [20.0, 5.0]) + e @ S.T
    out = responses(T, xy, e, excite)
    return [(round(angle_deg(d, r), 6) + 0.0, round(float(np.hypot(*r)), 6)) for d, r in out]


@pytest.mark.parametrize("deg", [0, 30, -100, 180])
def test_a_rotated_response_reads_its_angle_on_every_direction(deg):
    assert [abs(a) for a, _ in angles(rot(deg))] == [abs(deg)] * 2


def test_a_wrong_gain_keeps_the_angle():
    assert angles(rot(0, 0.29)) == [(0.0, 0.29)] * 2


def test_a_mirrored_axis_reads_180_along_it():
    got = sorted(abs(a) for a, _ in angles(np.diag([1.0, -1.0]), e=E * [0, 1]))
    assert got == [180.0]


def test_a_diagonal_push_is_one_direction_and_still_reads_right():
    diag = np.column_stack([E[:, 0], E[:, 0]])
    assert angles(np.eye(2), e=diag) == [(0.0, 1.0)]


def test_a_straight_line_push_says_nothing():
    assert angles(np.eye(2), e=np.column_stack([2.0 + 7.0 * T, -T])) == []


def test_a_tilt_held_long_pushes_at_roll_g_sin():
    c = load()["control"]
    t = np.linspace(1.0, 1.3, 31)
    e = pushed_mm([(0.0, 2.0, -1.0)], t, c)
    for k, deg in enumerate((2.0, -1.0)):
        acc = np.diff(e[:, k], 2) / 0.01**2
        assert acc == pytest.approx(ROLL * c["sim_g_mm_s2"] * math.sin(math.radians(deg)), rel=1e-3)


def feed(S, amp_deg=4.0, seconds=1.0):
    """Ball answers S times the push of a square wave tilt on x; returns the trip times."""
    cfg = load()
    watch, ticks = MappingWatch(cfg), np.arange(0.0, seconds, 0.01)
    cmds = [(t, amp_deg * (1 if math.sin(2 * math.pi * 2 * t) >= 0 else -1), 0.0) for t in ticks]
    xy = pushed_mm(cmds, ticks, cfg["control"]) @ np.asarray(S).T
    trips = []
    for i, t in enumerate(ticks):
        if watch.step(t, BallState(*xy[i], 0.0, 0.0, True, t), cmds[i][1:]):
            trips.append((round(t, 2), watch.bad))
    return trips


def test_a_flipped_push_trips_on_the_first_window():
    assert feed(rot(180)) == [(0.3, "x")]


def test_a_push_turned_past_the_line_needs_two_windows_in_a_row():
    assert feed(rot(100)) == [(0.4, "x")]


def test_the_right_push_never_trips_whatever_the_gain():
    assert feed(rot(0, 0.29)) == [] and feed(rot(30)) == [] and feed(rot(-60, 3.0)) == []


def test_a_ball_that_hardly_answers_is_not_judged():
    assert feed(rot(180, 0.1)) == []


def test_a_window_that_barely_tilts_is_not_judged():
    assert feed(rot(180), amp_deg=0.5) == []


def scripted(monkeypatch, script):
    # one canned answer per judged window: off is a response turned 100 deg, None judges nothing
    off = [(np.array([1.0, 0.0]), np.array([math.cos(1.75), math.sin(1.75)]))]
    answers = iter([off if s else [] for s in script])
    monkeypatch.setattr(mapping, "responses", lambda *a: next(answers, []))
    watch = MappingWatch(load())
    for i in range(31 + 10 * len(script)):
        t = i * 0.01
        watch.step(t, BallState(0.0, 0.0, 0.0, 0.0, True, t), (0.0, 0.0))
    return watch.bad


def test_two_off_windows_far_apart_are_not_in_a_row(monkeypatch):
    assert scripted(monkeypatch, [True, True]) == "x"
    assert scripted(monkeypatch, [True, None, None, True]) is None


def ticks(watch, valid, xs=lambda t: 0.0):
    for i in range(100):
        t = i * 0.01
        watch.step(t, BallState(xs(t), 0.0, 0.0, 0.0, valid(t), t), (0.0, 0.0))


def test_a_short_gap_keeps_the_window_a_long_one_starts_over():
    watch = MappingWatch(load())
    ticks(watch, lambda t: not 0.2 <= t < 0.35)
    assert watch.since == 0.0
    watch = MappingWatch(load())
    ticks(watch, lambda t: not 0.2 <= t < 0.5)
    assert watch.since == pytest.approx(0.5)


def test_a_sample_near_the_wall_starts_over():
    cfg = load()
    watch = MappingWatch(cfg)
    ticks(watch, lambda t: True, lambda t: 115.0 if 0.4 <= t < 0.45 else 0.0)
    assert watch.since == pytest.approx(0.45)
