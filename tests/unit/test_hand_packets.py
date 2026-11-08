import json
import socket
import time

import pytest

from src.contracts import Intent
from src.inputs.hand import HandInput, send_intent

pytestmark = pytest.mark.unit

GOOD = {"x": 0.1, "y": 0.2, "valid": True, "t_s": 1.0}
BAD = [
    b"\xff\x00 not json",
    b"[1, 2]",
    b'"x"',
    json.dumps({"y": 0.2, "valid": True, "t_s": 2.0}).encode(),
    json.dumps({**GOOD, "x": "0.5", "t_s": 2.0}).encode(),
    json.dumps({**GOOD, "t_s": "2.0"}).encode(),
    json.dumps({**GOOD, "t_s": 2.0, "seen": False, "calibrating": True, "boost": 1}).encode(),
]


def wait_for(hand_in, t_s):
    for _ in range(200):
        if hand_in.latest().t_s == t_s:
            return
        time.sleep(0.001)
    raise AssertionError(f"no intent at t_s={t_s}")


def test_bad_hand_packets_are_dropped_and_counted_and_a_good_one_still_lands():
    hand_in = HandInput(0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    to = ("127.0.0.1", hand_in.port)
    sender.sendto(json.dumps({**GOOD, "progress": 0.5, "lag_s": 0.02}).encode(), to)
    wait_for(hand_in, 1.0)
    for i, bad in enumerate(BAD, 1):
        sender.sendto(bad, to)
        for _ in range(200):
            if hand_in.latest() == Intent(**GOOD) and hand_in.dropped == i:
                break
            time.sleep(0.001)
        assert hand_in.dropped == i
        assert hand_in.latest() == Intent(**GOOD)
        assert hand_in.seen and not hand_in.calibrating
        assert hand_in.progress == 0.5 and hand_in.lag_s == 0.02
    send_intent(sender, Intent(0.3, 0.4, True, 3.0), hand_in.port, seen=True, calibrating=True)
    wait_for(hand_in, 3.0)
    assert hand_in.latest() == Intent(0.3, 0.4, True, 3.0)
    assert hand_in.calibrating and hand_in.lag_s is None
    assert hand_in.dropped == len(BAD)
    sender.close()
    hand_in.close()
