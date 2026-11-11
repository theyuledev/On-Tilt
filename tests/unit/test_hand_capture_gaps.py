import math

import pytest

from src.inputs.hand import HandIntent
from src.io.settings import load

pytestmark = pytest.mark.unit


def ready_hand(*, stale_s=0.25):
    cfg = load()["input"]
    hand = HandIntent(cfg, stale_s=stale_s)
    hand.update((0.0, 0.0), 0.0, recentre=True)
    for i in range(1, 61):
        before = hand.update((15.0, 0.0), i / 30)
    return hand, before


def test_capture_pause_restarts_smoothing_and_eases_in_without_a_missing_pose():
    hand, before = ready_hand()
    assert before.x > 0.5
    first = hand.update((-10.0, 0.0), 3.0)
    assert first.valid and (first.x, first.y) == (0.0, 0.0)
    assert hand.neutral == (0.0, 0.0)
    second = hand.update((-10.0, 0.0), 3.1)
    assert second.valid and second.x == pytest.approx(-0.4 / 3)


def test_long_capture_pause_requires_a_fresh_still_neutral_window():
    hand, _ = ready_hand()
    first = hand.update((15.0, 0.0), 10.0)
    assert not first.valid and hand.calibrating and hand.progress == 0.0
    assert hand.neutral is None
    for i in range(1, 31):
        out = hand.update((15.0, 0.0), 10.0 + i / 30)
    assert out.valid and (out.x, out.y) == (0.0, 0.0)
    assert hand.neutral == (15.0, 0.0)


def test_capture_pause_cannot_count_unseen_time_toward_neutral():
    hand = HandIntent(load()["input"], stale_s=0.25)
    for t in (0.0, 0.1, 0.2):
        assert not hand.update((0.0, 0.0), t).valid
    assert not hand.update((0.0, 0.0), 0.6).valid
    assert hand.progress == 0.0
    for i in range(1, 7):
        assert not hand.update((0.0, 0.0), 0.6 + i / 10).valid
    for i in range(7, 12):
        out = hand.update((0.0, 0.0), 0.6 + i / 10)
    assert out.valid and hand.neutral == (0.0, 0.0)


def test_recentre_after_a_long_capture_pause_still_sets_neutral_immediately():
    hand, _ = ready_hand()
    out = hand.update((20.0, -10.0), 10.0, recentre=True)
    assert out.valid and (out.x, out.y) == (0.0, 0.0)
    assert hand.neutral == (20.0, -10.0)
    assert hand.update((20.0, -10.0), 10.1).x == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_capture_stamp_is_invalid_without_poisoning_the_next_pose(bad):
    hand, before = ready_hand()
    out = hand.update((-25.0, 25.0), bad)
    assert not out.valid and (out.x, out.y) == (0.0, 0.0)
    assert math.isfinite(out.t_s)
    recovered = hand.update((15.0, 0.0), 2.1)
    assert recovered.valid and recovered.x == pytest.approx(before.x, abs=1e-5)
    assert recovered.y == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_bad_first_capture_stamp_does_not_begin_neutral_calibration(bad):
    hand = HandIntent(load()["input"], stale_s=0.25)
    out = hand.update((0.0, 0.0), bad)
    assert not out.valid and math.isfinite(out.t_s)
    assert not hand.seen and not hand.calibrating and hand.progress == 0.0
    assert not hand.update((0.0, 0.0), 1.0).valid
    assert hand.calibrating and hand.progress == 0.0


def test_old_constructors_use_the_existing_neutral_window_for_capture_gaps():
    hand, before = ready_hand(stale_s=None)
    out = hand.update((15.0, 0.0), 2.5)
    assert out.valid and out.x == pytest.approx(before.x, abs=1e-5)


def test_a_mode_switch_reads_like_a_hand_built_in_the_new_mode():
    tilt = {**load()["input"], "hand_mode": "tilt"}
    hand = HandIntent(tilt)
    for i in range(3):
        hand.update((15.0, 0.0), i / 30)
    hand.set_mode("palm")
    fresh = HandIntent({**tilt, "hand_mode": "palm"})
    times = [(2 + i) / 30 for i in range(45)]

    def hold(h):
        return [(h.update((0.5, 0.5), t), h.progress) for t in times]

    # no degrees left in the median and no old capture stamp holding frames back
    assert hold(hand) == hold(fresh)
    assert hand.neutral == fresh.neutral == pytest.approx((0.5, 0.5))
