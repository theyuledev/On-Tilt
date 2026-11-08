import json
import socket
from dataclasses import asdict

import pytest

from src.app import Loop
from src.contracts import Intent
from src.inputs import hand_process
from src.inputs.hand import HandInput, HandIntent, recentre_asked, send_intent
from src.io.camera import CameraSource
from src.io.settings import load
from tests.unit.test_hand import CFG, CountingCapture, FakeHand, calibrated

pytestmark = pytest.mark.unit


def test_neutral_survives_a_short_dropout():
    hand, t = calibrated()
    assert not hand.update(None, t + 0.5).valid
    assert hand.update(None, t + 3.0).valid is False and hand.neutral is not None
    assert hand.progress == 1.0
    # the palm comes back and steers straight away, no hold still
    assert hand.update((0.5 + 0.125, 0.5), t + 3.1).x == pytest.approx(0.5)


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
    hand.update((0.3 + 0.125, 0.6), 0.1)
    assert hand.update((0.4, 0.4), 0.2, recentre=True) == Intent(0.0, 0.0, True, 0.2)
    assert hand.neutral == pytest.approx((0.4, 0.4))


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


def wait_for(hand_in: HandInput, t_s: float) -> None:
    for _ in range(100):
        if hand_in.latest().t_s == t_s:
            return


def test_progress_rides_the_udp_message_and_recentre_goes_back_the_same_way():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(("127.0.0.1", 0))
    sender.setblocking(False)
    hand_in.recentre()  # nobody to tell yet
    assert not recentre_asked(sender)
    send_intent(
        sender,
        Intent(0.0, 0.0, False, 1.0),
        hand_in.port,
        seen=True,
        calibrating=True,
        progress=0.5,
    )
    wait_for(hand_in, 1.0)
    assert hand_in.progress == 0.5
    hand_in.recentre()
    for _ in range(100):
        if recentre_asked(sender):
            break
    else:
        pytest.fail("the hand process never heard the re centre")
    assert not recentre_asked(sender)
    # older senders have no progress field
    sender.sendto(
        json.dumps(asdict(Intent(0.0, 0.0, True, 2.0))).encode(), ("127.0.0.1", hand_in.port)
    )
    wait_for(hand_in, 2.0)
    assert hand_in.progress == 0.0
    sender.close()
    hand_in.close()


def test_hand_process_recentres_on_the_frame_the_key_arrives():
    cap = CountingCapture()
    hand = HandIntent(CFG)
    asked = [False, True]
    sent = []

    def model(img, t):
        if not asked:
            cap.alive = False
        return (0.2, 0.7)

    def recentre():
        return asked.pop() if asked else False

    frames = CameraSource(cap)
    hand_process.run(frames, model, hand, lambda intent, **flags: sent.append(flags), recentre)
    frames.close()
    assert hand.neutral == (0.2, 0.7)
    assert all(f["progress"] == 1.0 for f in sent)


def test_snapshot_carries_hand_progress():
    cfg = load()
    assert Loop(cfg).snapshot()["hand_progress"] == 0.0
    loop = Loop(cfg, FakeHand(Intent(0.0, 0.0, False, 10.0), calibrating=True, progress=0.4))
    loop.tick(10.01, 0.01)
    assert loop.snapshot()["hand_progress"] == 0.4
