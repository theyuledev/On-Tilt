import copy
import math
from types import SimpleNamespace

import pytest

from src.contracts import Intent
from src.inputs.hand import HandIntent
from src.inputs.hand_process import run
from src.inputs.prediction import HandPredictor
from src.io.settings import BASE, load

pytestmark = pytest.mark.unit
DT = 1 / 30


@pytest.fixture
def cfg():
    return load(local=BASE)


def new_hand(cfg):
    inp = {**cfg["input"], "hand_mode": "tilt"}
    return HandIntent(inp, stale_s=cfg["rig"]["stale_intent_s"])


def ready_hand(cfg):
    hand = new_hand(cfg)
    hand.update((0.0, 0.0), 0.0, recentre=True)
    at = (0.6 * hand.span, 0.0)
    for i in range(1, 61):
        hand.update(at, i * DT)
    return hand, 60 * DT, at


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("back", [0.0, DT / 2])
def test_old_or_duplicate_recentre_cannot_change_state_even_after_a_newer_loss(cfg, missing, back):
    hand, latest, at = ready_hand(cfg)
    if missing:
        latest += DT
        hand.update(None, latest)
    before = copy.deepcopy(hand.__dict__)
    out = hand.update((-hand.span, hand.span), latest - back, recentre=True)
    assert not out.valid and (out.x, out.y) == (0.0, 0.0)
    assert math.isfinite(out.t_s) and hand.__dict__ == before
    recovered = hand.update(at, latest + DT)
    assert recovered.valid and hand.neutral == (0.0, 0.0)


@pytest.mark.parametrize("recentre", [False, True])
def test_rejected_capture_cannot_add_neutral_samples_or_lock_neutral(cfg, recentre):
    hand = new_hand(cfg)
    hand.update((0.0, 0.0), 0.0)
    hand.update((0.0, 0.0), DT)
    before = copy.deepcopy(hand.__dict__)
    out = hand.update((hand.span, hand.span), DT / 2, recentre=recentre)
    assert not out.valid and hand.neutral is None
    assert hand.__dict__ == before
    for i in range(2, math.ceil(cfg["input"]["neutral_s"] / DT) + 2):
        out = hand.update((0.0, 0.0), i * DT)
    assert out.valid and hand.neutral == (0.0, 0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("started", [False, True])
def test_nonfinite_capture_time_cannot_advance_or_mutate_the_hand(cfg, bad, started):
    hand, latest, at = ready_hand(cfg) if started else (new_hand(cfg), 0.0, (0.0, 0.0))
    before = copy.deepcopy(hand.__dict__)
    out = hand.update((-hand.span, hand.span), bad, recentre=True)
    assert not out.valid and math.isfinite(out.t_s)
    assert hand.__dict__ == before
    recovered = hand.update(at, latest + DT)
    assert recovered.valid == started
    assert all(math.isfinite(v) for v in (recovered.x, recovered.y, recovered.t_s))


@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("fps", [15, 30, 60])
def test_first_command_sent_after_a_silent_capture_gap_is_zero(cfg, axis, sign, fps):
    dt = 1 / fps
    hand = new_hand(cfg)
    hand.update((0.0, 0.0), 0.0, recentre=True)
    times = [i * dt for i in range(1, 64)]
    times.append(times[-1] + hand.stale_s + dt)
    readings = []
    for i in range(len(times)):
        xy = [0.0, 0.0]
        xy[axis] = sign * hand.span * (0.6 if i < 60 else 0.3)
        readings.append(tuple(xy))
    pending = [SimpleNamespace(image=at, t_s=t) for at, t in zip(readings, times, strict=True)]
    published = iter(t + dt for t in times)
    sent = []

    class Frames:
        @property
        def running(self):
            return bool(pending)

        def read(self):
            return pending.pop(0)

    run(
        Frames(),
        lambda image, t: image,
        hand,
        lambda intent, **flags: sent.append(intent),
        clock=lambda: next(published),
    )
    attr = ("x", "y")[axis]
    assert 0.1 < abs(getattr(sent[-2], attr)) < abs(getattr(sent[-4], attr))
    assert sent[-1].valid and sent[-1].t_s == times[-1]
    assert (sent[-1].x, sent[-1].y) == (0.0, 0.0)


def test_predictor_discards_pre_gap_velocity_and_then_predicts_only_the_new_ramp(cfg):
    stale_s = cfg["rig"]["stale_intent_s"]
    dt = stale_s / 3
    predictor = HandPredictor(stale_s=stale_s)
    for x, t in ((0.6, 0.0), (0.3, dt)):
        predictor.update(Intent(x, 0.0, True, t), t + dt / 2)
    resume = dt + stale_s + DT
    out = predictor.update(Intent(0.0, 0.0, True, resume), resume + dt / 2)
    assert out.valid and (out.x, out.y) == (0.0, 0.0)
    for i in (1, 2):
        out = predictor.update(Intent(i / 10, 0.0, True, resume + i * dt), resume + i * dt + dt / 2)
        assert out.x >= i / 10
    assert out.x == pytest.approx(0.25)


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_prediction_still_works_at_or_below_the_capture_gap_threshold(cfg, fraction):
    stale_s = cfg["rig"]["stale_intent_s"]
    dt = stale_s * fraction
    predictor = HandPredictor(stale_s=stale_s)
    for i in range(3):
        out = predictor.update(Intent(i / 10, 0.0, True, i * dt), i * dt + dt / 2)
    assert out.valid and out.x == pytest.approx(0.25)
