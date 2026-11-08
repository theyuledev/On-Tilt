import json
import socket
import time
from dataclasses import asdict
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from src.app import Loop
from src.contracts import Intent
from src.inputs import hand_process
from src.inputs.hand import (
    PALM,
    HandInput,
    HandIntent,
    palm_center,
    recentre_asked,
    send_intent,
)
from src.io.camera import CameraSource
from src.io.settings import load

pytestmark = pytest.mark.unit

CFG = {"hand_span": 0.25, "neutral_s": 1.0, "neutral_keep_s": 4.0, "smoothing": 0.0}


def calibrated(cfg=CFG, at=(0.5, 0.5)):
    hand = HandIntent(cfg)
    t = 0.0
    while hand.neutral is None:
        hand.update(at, t)
        t += 0.1
    return hand, t


def test_palm_center_is_the_mean_of_wrist_and_finger_bases():
    pts = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    for i in PALM:
        pts[i] = SimpleNamespace(x=0.2 * i / 17, y=0.6)
    x, y = palm_center(pts)
    assert x == pytest.approx(sum(0.2 * i / 17 for i in PALM) / 5)
    assert y == pytest.approx(0.6)


def test_no_intent_until_a_second_of_neutral():
    hand = HandIntent(CFG)
    assert not hand.update((0.5, 0.5), 0.0).valid
    assert not hand.update((0.5, 0.5), 0.5).valid
    assert hand.update((0.5, 0.5), 1.0).valid


def test_a_sweeping_hand_never_sets_neutral_but_a_still_one_does():
    hand = HandIntent(CFG)
    step = CFG["hand_span"] / 4
    for i in range(40):
        assert not hand.update((0.1 + step * i / 4, 0.5), i * 0.1).valid
    assert hand.neutral is None
    assert hand.calibrating
    t = 4.0
    while hand.neutral is None:
        hand.update((0.3, 0.6), t)
        t += 0.1
    assert hand.neutral == pytest.approx((0.3, 0.6))
    assert t < 6.0  # the window holding the tail of the sweep is dropped, the next one locks
    assert not hand.calibrating


def test_a_hand_moving_less_than_a_quarter_span_still_locks():
    hand = HandIntent(CFG)
    wobble = CFG["hand_span"] / 4 * 0.9
    for i in range(11):
        hand.update((0.5 + wobble * (i % 2), 0.5), i * 0.1)
    assert hand.neutral is not None


def test_seen_and_calibrating_follow_the_palm():
    hand = HandIntent(CFG)
    assert not hand.seen and not hand.calibrating
    hand.update((0.5, 0.5), 0.0)
    assert hand.seen and hand.calibrating
    hand.update(None, 0.1)
    assert not hand.seen and not hand.calibrating
    hand, _ = calibrated()
    assert hand.seen and not hand.calibrating


def test_moving_your_hand_right_and_up_is_plus_x_plus_y():
    hand, t = calibrated()
    i = hand.update((0.5 - 0.125, 0.5 - 0.125), t)
    assert i.x == pytest.approx(0.5) and i.y == pytest.approx(0.5)


def test_intent_clamps_at_one():
    hand, t = calibrated()
    i = hand.update((0.0, 1.0), t)
    assert (i.x, i.y) == (1.0, -1.0)


def test_smoothing_eases_in():
    hand, t = calibrated({**CFG, "smoothing": 0.5})
    first = hand.update((0.25, 0.5), t).x
    second = hand.update((0.25, 0.5), t + 0.03).x
    assert first == pytest.approx(0.5) and second == pytest.approx(0.75)


def test_neutral_survives_a_short_dropout():
    hand, t = calibrated()
    assert not hand.update(None, t + 0.5).valid
    assert hand.update(None, t + 3.0).valid is False and hand.neutral is not None
    assert hand.progress == 1.0
    # the palm comes back and steers straight away, no hold still
    assert hand.update((0.5 - 0.125, 0.5), t + 3.1).x == pytest.approx(0.5)


def test_neutral_drops_after_a_long_dropout():
    hand, t = calibrated()
    hand.update(None, t + 5.0)
    assert hand.neutral is None and hand.progress == 0.0
    assert not hand.update((0.3, 0.3), t + 5.1).valid
    assert hand.calibrating


def test_a_dropout_longer_than_the_window_restarts_the_hold_still():
    hand = HandIntent(CFG)
    hand.update((0.5, 0.5), 0.0)
    hand.update((0.5, 0.5), 0.5)
    hand.update(None, 1.6)
    assert hand.progress == 0.0
    assert not hand.update((0.5, 0.5), 1.7).valid


def test_recentre_sets_neutral_to_the_palm_at_once():
    hand = HandIntent(CFG)
    i = hand.update((0.3, 0.6), 0.0, recentre=True)
    assert i.valid and (i.x, i.y) == (0.0, 0.0)
    assert hand.neutral == (0.3, 0.6) and hand.progress == 1.0
    hand.update((0.3 - 0.125, 0.6), 0.1)
    assert hand.update((0.4, 0.4), 0.2, recentre=True) == Intent(0.0, 0.0, True, 0.2)
    assert hand.neutral == (0.4, 0.4)


def test_recentre_without_a_palm_does_nothing():
    hand = HandIntent(CFG)
    assert not hand.update(None, 0.0, recentre=True).valid
    assert hand.neutral is None and not hand.calibrating
    hand, t = calibrated()
    hand.update(None, t + 0.1, recentre=True)
    assert hand.neutral == pytest.approx((0.5, 0.5))


def test_progress_tracks_the_hold_still_window():
    hand = HandIntent(CFG)
    assert hand.progress == 0.0
    hand.update((0.5, 0.5), 0.0)
    assert hand.progress == 0.0
    hand.update((0.5, 0.5), 0.5)
    assert hand.progress == pytest.approx(0.5)
    hand.update((0.5, 0.5), 0.9)
    assert hand.progress == pytest.approx(0.9)
    hand.update((0.5, 0.5), 1.0)
    assert hand.progress == 1.0 and hand.neutral is not None


def test_progress_restarts_when_a_moving_hand_throws_the_window_away():
    hand = HandIntent(CFG)
    hand.update((0.1, 0.5), 0.0)
    hand.update((0.5, 0.5), 0.5)
    assert hand.progress == pytest.approx(0.5)
    hand.update((0.9, 0.5), 1.0)
    assert hand.progress == 0.0 and hand.neutral is None


def test_socket_round_trip():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_intent(sender, Intent(0.1, -0.2, True, 5.0), hand_in.port, seen=True, calibrating=True)
    send_intent(
        sender,
        Intent(0.3, 0.4, True, 6.0),
        hand_in.port,
        seen=True,
        calibrating=False,
        progress=0.5,
        lag_s=0.03,
    )
    for _ in range(100):
        if hand_in.latest().t_s == 6.0:
            break
    assert hand_in.latest() == Intent(0.3, 0.4, True, 6.0)
    assert hand_in.seen and not hand_in.calibrating
    assert hand_in.progress == 0.5
    assert hand_in.lag_s == 0.03
    sender.close()
    hand_in.close()


def test_recentre_goes_back_to_the_hand_process_once_it_has_spoken():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(("127.0.0.1", 0))
    sender.setblocking(False)
    hand_in.recentre()  # nobody to tell yet
    assert not recentre_asked(sender)
    send_intent(sender, Intent(0.0, 0.0, False, 1.0), hand_in.port, seen=True, calibrating=True)
    for _ in range(100):
        if hand_in.latest().t_s == 1.0:
            break
    hand_in.recentre()
    for _ in range(100):
        if recentre_asked(sender):
            break
    else:
        pytest.fail("the hand process never heard the re centre")
    assert not recentre_asked(sender)
    sender.close()
    hand_in.close()


def test_an_intent_without_the_flags_still_counts_as_a_seen_hand():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    msg = json.dumps(asdict(Intent(0.3, 0.4, True, 6.0))).encode()
    sender.sendto(msg, ("127.0.0.1", hand_in.port))
    for _ in range(100):
        if hand_in.latest().t_s == 6.0:
            break
    assert hand_in.latest() == Intent(0.3, 0.4, True, 6.0)
    assert hand_in.seen and not hand_in.calibrating
    assert hand_in.progress == 0.0 and hand_in.lag_s is None
    sender.close()
    hand_in.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_hand_intent_is_dropped_and_the_last_one_goes_stale(bad):
    cfg = load()
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    good = Intent(0.5, 0.0, True, 10.0)
    send_intent(sender, good, hand_in.port, seen=True, calibrating=False)
    send_intent(sender, Intent(bad, 0.0, True, 10.5), hand_in.port, seen=False, calibrating=True)
    send_intent(sender, Intent(0.0, bad, True, 10.5), hand_in.port, seen=False, calibrating=True)
    for _ in range(100):
        if hand_in.latest().t_s == 10.0:
            break
    for _ in range(20):
        time.sleep(0.005)
        assert hand_in.latest() == good
        assert hand_in.seen and not hand_in.calibrating

    loop = Loop(cfg, hand_in)
    stale_s = cfg["rig"]["stale_intent_s"]
    loop.keys._intent = Intent(-1.0, 0.0, True, 10.0 + stale_s)
    assert loop.tick(10.01, 0.01).intent.x == 0.5
    assert loop.tick(10.01 + stale_s, 0.01).intent.x == -1.0
    sender.close()
    hand_in.close()


class CountingCapture:
    """Frames carry their number, a new one every millisecond."""

    def __init__(self):
        self.i = 0
        self.alive = True

    def read(self):
        if not self.alive:
            return False, None
        self.i += 1
        time.sleep(0.001)
        return True, np.full((2, 2), self.i, np.int32)

    def release(self):
        pass


def test_hand_process_runs_the_model_on_the_newest_frame_and_reports_lag():
    cap = CountingCapture()
    seen, sent = [], []

    def slow_model(img, t):
        seen.append(int(img[0, 0]))
        time.sleep(0.02)
        if len(seen) == 5:
            cap.alive = False
        return None

    def send(intent, **flags):
        sent.append(flags)

    frames = CameraSource(cap)
    hand_process.run(frames, slow_model, HandIntent(CFG), send)
    frames.close()
    assert len(seen) >= 5 and len(sent) == len(seen)
    assert all(b > a for a, b in pairwise(seen))
    # the frames that piled up while the model ran were skipped, not queued
    assert max(b - a for a, b in pairwise(seen)) > 1
    assert all(f["lag_s"] >= 0 and f["seen"] is False and f["progress"] == 0.0 for f in sent)


def test_hand_process_recentres_on_the_frame_the_key_arrives():
    cap = CountingCapture()
    hand = HandIntent(CFG)
    asked = [False, True]

    def model(img, t):
        if not asked:
            cap.alive = False
        return (0.2, 0.7)

    def recentre():
        return asked.pop() if asked else False

    frames = CameraSource(cap)
    hand_process.run(frames, model, hand, lambda intent, **flags: None, recentre)
    frames.close()
    assert hand.neutral == (0.2, 0.7)


class FakeHand:
    def __init__(self, intent, seen=True, calibrating=False, progress=0.0):
        self.intent = intent
        self.seen = seen
        self.calibrating = calibrating
        self.progress = progress

    def latest(self):
        return self.intent

    def recentre(self):
        pass


def test_loop_uses_a_fresh_hand_and_falls_back_to_keys():
    cfg = load()
    loop = Loop(cfg, FakeHand(Intent(0.5, 0.0, True, 10.0)))
    loop.keys._intent = Intent(-1.0, 0.0, True, 10.0)
    assert loop.tick(10.01, 0.01).intent.x == 0.5

    loop.hand.intent = Intent(0.5, 0.0, True, 9.0)
    assert loop.tick(10.02, 0.01).intent.x == -1.0

    loop.hand.intent = Intent(0.5, 0.0, False, 10.02)
    assert loop.tick(10.03, 0.01).intent.x == -1.0


def test_snapshot_carries_the_hand_flags():
    cfg = load()
    snap = Loop(cfg).snapshot()
    assert snap["hand_seen"] is False and snap["hand_calibrating"] is False
    assert snap["hand_progress"] == 0.0
    hand = FakeHand(Intent(0.0, 0.0, False, 10.0), seen=True, calibrating=True, progress=0.4)
    loop = Loop(cfg, hand)
    loop.tick(10.01, 0.01)
    snap = loop.snapshot()
    assert snap["hand_seen"] and snap["hand_calibrating"] and snap["hand_progress"] == 0.4
    loop.hand.calibrating = False
    loop.tick(10.02, 0.01)
    assert not loop.snapshot()["hand_calibrating"]


def test_hand_flags_drop_when_the_hand_process_goes_quiet():
    cfg = load()
    loop = Loop(cfg, FakeHand(Intent(0.5, 0.0, True, 10.0), seen=True, calibrating=True))
    loop.tick(10.01, 0.01)
    assert loop.snapshot()["hand_seen"] and loop.snapshot()["hand_calibrating"]
    stale_s = cfg["rig"]["stale_intent_s"]
    loop.tick(9.99 + stale_s, 0.01)
    assert loop.snapshot()["hand_seen"]
    loop.tick(10.01 + stale_s, 0.01)
    assert not loop.snapshot()["hand_seen"] and not loop.snapshot()["hand_calibrating"]
