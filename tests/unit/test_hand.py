import socket
from types import SimpleNamespace

import pytest

from src.app import Loop
from src.contracts import Intent
from src.inputs.hand import PALM, HandInput, HandIntent, palm_center, send_intent
from src.io.settings import load

pytestmark = pytest.mark.unit

CFG = {"hand_span": 0.25, "neutral_s": 1.0, "smoothing": 0.0}


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


def test_neutral_resets_after_a_second_without_a_hand():
    hand, t = calibrated()
    assert not hand.update(None, t + 0.5).valid
    assert hand.neutral is not None
    hand.update(None, t + 1.5)
    assert hand.neutral is None


def test_socket_round_trip():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_intent(sender, Intent(0.1, -0.2, True, 5.0), hand_in.port)
    send_intent(sender, Intent(0.3, 0.4, True, 6.0), hand_in.port)
    for _ in range(100):
        if hand_in.latest().t_s == 6.0:
            break
    assert hand_in.latest() == Intent(0.3, 0.4, True, 6.0)
    sender.close()
    hand_in.close()


class FakeHand:
    def __init__(self, intent):
        self.intent = intent

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
