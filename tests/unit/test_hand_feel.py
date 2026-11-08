import numpy as np
import pytest

from src.inputs.hand import HandIntent
from src.inputs.one_euro import one_euro
from src.inputs.shaping import floor_deg, shape, top_deg
from src.io.settings import load

pytestmark = pytest.mark.unit

CFG = load()
INPUT = CFG["input"]
TILT = {**INPUT, "hand_mode": "tilt", "hand_tilt_span_deg": 25.0}
DT = 1 / 30
SPAN = TILT["hand_tilt_span_deg"]
JITTER_DEG = 1.5


def filtered(xs):
    state, out = None, []
    for i, x in enumerate(xs):
        y, state = one_euro(state, x, i * DT, INPUT)
        out.append(y)
    return np.array(out)


def test_one_euro_passes_a_constant_through():
    assert (filtered([0.4] * 50) == 0.4).all()


def test_one_euro_keeps_its_output_when_time_does_not_move():
    _, s = one_euro(None, 0.0, 0.0, INPUT)
    y, s = one_euro(s, 1.0, 0.1, INPUT)
    assert one_euro(s, 5.0, 0.1, INPUT) == (y, s)
    assert one_euro(s, 5.0, 0.05, INPUT) == (y, s)


def test_one_euro_matches_the_first_two_steps_by_hand():
    cfg = {"hand_min_cutoff_hz": 1.0, "hand_beta": 0.5, "hand_d_cutoff_hz": 2.0}
    y, s = one_euro(None, 0.0, 0.0, cfg)
    assert (y, s) == (0.0, (0.0, 0.0, 0.0))
    y, s = one_euro(s, 1.0, 0.1, cfg)
    # d: tau = 1 / 4pi = 0.07958, a = 1 / (1 + 0.7958) = 0.55686, dx = 0.55686 * 10
    # x: fc = 1 + 0.5 * 5.5686 = 3.7843, tau = 0.042057, a = 1 / (1 + 0.42057) = 0.70395
    assert s[1] == pytest.approx(5.5686, abs=1e-4)
    assert y == pytest.approx(0.70395, abs=1e-5)
    assert s == (y, s[1], 0.1)


def test_a_still_hand_shakes_less_than_with_the_old_ema():
    raw = np.random.default_rng(243).normal(0.0, JITTER_DEG, 300) / SPAN
    ema, e = [], raw[0]
    for x in raw:
        e = 0.3 * e + 0.7 * x
        ema.append(e)
    settled = slice(30, None)

    def ratio(out):
        return np.std(np.asarray(out)[settled]) / np.std(raw[settled])

    assert ratio(filtered(raw)) <= 0.6
    assert ratio(ema) >= 0.6


def test_a_quick_tilt_lands_within_100_ms():
    to = 20 / SPAN
    out = filtered([0.0] * 30 + [to] * 30)
    t90 = DT * next(i for i in range(30, 60) if out[i] >= 0.9 * to) - DT * 30
    assert t90 <= 0.1


def test_a_steady_sweep_lags_under_40_ms():
    v = 60 / SPAN
    n = round(0.8 / DT)
    xs = np.r_[np.full(30, -24 / SPAN), -24 / SPAN + v * DT * np.arange(n + 1)]
    out = filtered(xs)
    second_half = slice(30 + n // 2, 30 + n + 1)
    assert np.mean((xs - out)[second_half] / v) <= 0.04


def shaky_hand(seed, seconds):
    rng = np.random.default_rng(seed)
    for i in range(round(seconds / DT)):
        yield i * DT, (5.0 + rng.normal(0.0, JITTER_DEG), -10.0 + rng.normal(0.0, JITTER_DEG))


def test_a_shaky_still_hand_locks_neutral_within_2_s():
    hand = HandIntent(TILT)
    for t, reading in shaky_hand(243, 3.0):
        hand.update(reading, t)
        if hand.neutral is not None:
            break
    assert hand.neutral is not None and t <= 2.0


def test_recentre_on_a_shaky_hand_keeps_the_plate_level():
    hand = HandIntent(TILT)
    cmds = []
    for i, (t, reading) in enumerate(shaky_hand(243, 5.0)):
        intent = hand.update(reading, t, recentre=i == 60)
        if i >= 60:
            cmds.append(shape(intent.x, intent.y, CFG))
    assert cmds[0] == (0.0, 0.0)
    # jitter can poke past the dead zone edge now and then: that lands on the friction
    # floor, which holds a resting ball, and the smooth edge keeps the rest tiny
    (fx, fy), (tx, ty) = floor_deg(CFG), top_deg(CFG)
    assert max(abs(c[0]) for c in cmds) < fx + 0.05 * tx
    assert max(abs(c[1]) for c in cmds) < fy + 0.05 * ty


RAW_TILT = {**TILT, "hand_min_cutoff_hz": 1e9, "hand_beta": 0.0}  # filter passes through
RAMP_S = INPUT["key_ramp_s"]


def eases_in(hand, reading, t0, full):
    # frames after t0, the moment the hand showed up or the neutral moved
    for i in range(1, round(2 * RAMP_S / DT)):
        x = hand.update(reading, t0 + i * DT).x
        part = min(1.0, i * DT / RAMP_S)
        assert abs(x) <= full * part + 1e-6
        if part == 1.0:
            assert x == pytest.approx(full)


def test_no_jump_when_the_hand_comes_back_tilted():
    hand = HandIntent(RAW_TILT)
    hand.update((0.0, 0.0), 0.0, recentre=True)
    hand.update((0.0, 0.0), 0.5)
    assert not hand.update(None, 0.6).valid
    i = hand.update((20.0, 0.0), 1.0)
    assert i.valid and i.x == 0.0
    eases_in(hand, (20.0, 0.0), 1.0, 20 / SPAN)


def test_no_ease_in_when_key_ramp_s_is_zero():
    hand = HandIntent({**RAW_TILT, "key_ramp_s": 0.0})
    hand.update((0.0, 0.0), 0.0, recentre=True)
    assert not hand.update(None, 0.5).valid
    assert hand.update((20.0, 0.0), 1.0).x == pytest.approx(20 / SPAN)


def test_the_ease_in_starts_over_on_recentre():
    hand = HandIntent(RAW_TILT)
    hand.update((0.0, 0.0), 0.0, recentre=True)
    assert hand.update((20.0, 0.0), 0.5).x == pytest.approx(20 / SPAN)
    assert hand.update((-20.0, 0.0), 1.0, recentre=True).x == 0.0
    eases_in(hand, (0.0, 0.0), 1.0, 20 / SPAN)
