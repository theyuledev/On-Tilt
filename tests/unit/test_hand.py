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
from src.inputs.hand import PALM, HandInput, HandIntent, palm_center, send_intent
from src.io.camera import CameraSource
from src.io.settings import load

pytestmark = pytest.mark.unit

CFG = {
    "hand_span": 0.25,
    "hand_tilt_span_deg": 25.0,
    "neutral_s": 1.0,
    "neutral_keep_s": 4.0,
    "hand_mode": "palm",
    "hand_flip_x": False,
    "hand_flip_y": False,
    # a filter that passes straight through
    "hand_min_cutoff_hz": 1e9,
    "hand_beta": 0.0,
    "hand_d_cutoff_hz": 1.0,
}


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
    # the webcam looks down from above, so the player's right is image right
    hand, t = calibrated()
    i = hand.update((0.5 + 0.125, 0.5 - 0.125), t)
    assert i.x == pytest.approx(0.5) and i.y == pytest.approx(0.5)


def test_intent_clamps_at_one():
    hand, t = calibrated()
    i = hand.update((1.0, 1.0), t)
    assert (i.x, i.y) == (1.0, -1.0)


def test_socket_round_trip():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_intent(sender, Intent(0.1, -0.2, True, 5.0), hand_in.port, seen=True, calibrating=True)
    send_intent(
        sender, Intent(0.3, 0.4, True, 6.0), hand_in.port, seen=True, calibrating=False, lag_s=0.03
    )
    for _ in range(100):
        if hand_in.latest().t_s == 6.0:
            break
    assert hand_in.latest() == Intent(0.3, 0.4, True, 6.0)
    assert hand_in.seen and not hand_in.calibrating
    assert hand_in.lag_s == 0.03
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
    assert hand_in.lag_s is None
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
    assert all(f["lag_s"] >= 0 and f["seen"] is False for f in sent)


class FakeHand:
    def __init__(self, intent, seen=True, calibrating=False, progress=0.0):
        self.intent = intent
        self.seen = seen
        self.calibrating = calibrating
        self.progress = progress

    def latest(self):
        return self.intent


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
    loop = Loop(cfg, FakeHand(Intent(0.0, 0.0, False, 10.0), seen=True, calibrating=True))
    loop.tick(10.01, 0.01)
    snap = loop.snapshot()
    assert snap["hand_seen"] and snap["hand_calibrating"]
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
