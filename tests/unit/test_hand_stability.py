import math

import pytest

from src.inputs.hand import HandIntent
from src.io.settings import load

pytestmark = pytest.mark.unit

DT = 1 / 30


@pytest.fixture
def hand():
    cfg = load()["input"]
    tracker = HandIntent(cfg)
    tracker.update((0.0, 0.0), 0.0, recentre=True)
    hold(tracker, (0.0, 0.0), 0.0)
    return tracker


def hold(hand, reading, t, seconds=1.0):
    intents = []
    for _ in range(round(seconds / DT)):
        t += DT
        intents.append(hand.update(reading, t))
    return t, intents


def reading(hand, x, y):
    key = "hand_tilt_span_deg" if hand.cfg["hand_mode"] == "tilt" else "hand_span"
    return x * hand.cfg[key], y * hand.cfg[key]


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_bad_pose_never_becomes_a_full_tilt_and_a_good_pose_recovers(hand, axis, bad):
    at = list(reading(hand, 0.3, -0.2))
    t, _ = hold(hand, tuple(at), 1.0)
    at[axis] = bad
    lost = hand.update(tuple(at), t + DT)
    assert not lost.valid and (lost.x, lost.y) == (0.0, 0.0)
    t, recovered = hold(hand, reading(hand, 0.3, -0.2), t + DT)
    assert all(i.valid and math.isfinite(i.x + i.y) for i in recovered)
    assert (recovered[-1].x, recovered[-1].y) == pytest.approx((0.3, 0.2), abs=0.01)


def test_recentring_during_motion_keeps_a_held_pose_at_zero(hand):
    at = reading(hand, 0.8, -0.6)
    hand.update(at, 1.0 + DT)
    centred = hand.update(at, 1.0 + 2 * DT, recentre=True)
    _, held = hold(hand, at, 1.0 + 2 * DT)
    assert centred.valid and (centred.x, centred.y) == (0.0, 0.0)
    assert all(i.valid and abs(i.x) < 1e-6 and abs(i.y) < 1e-6 for i in held)


def test_one_bad_pose_does_not_kick_a_steady_off_neutral_tilt(hand):
    at = reading(hand, 0.35, -0.2)
    t, held = hold(hand, at, 1.0, seconds=2.0)
    before = held[-1]
    spike = hand.update(reading(hand, -0.9, 0.9), t + DT)
    _, after = hold(hand, at, t + DT, seconds=0.5)
    for intent in (spike, *after):
        assert intent.valid
        assert (intent.x, intent.y) == pytest.approx((before.x, before.y), abs=0.005)


@pytest.mark.parametrize("axis", [0, 1], ids=["left_right", "forward_back"])
@pytest.mark.parametrize("first_sign", [-1, 1])
def test_sustained_movement_and_reversal_keep_the_full_useful_range(hand, axis, first_sign):
    t = 1.0
    for sign in (first_sign, -first_sign):
        at = [0.0, 0.0]
        at[axis] = sign * 0.8
        t, moved = hold(hand, reading(hand, *at), t)
        x, y = moved[-1].x, moved[-1].y
        expected = sign if axis == 0 else -sign
        assert moved[-1].valid
        assert expected * (x, y)[axis] > 0.75
        assert abs((x, y)[1 - axis]) < 1e-6
