from itertools import pairwise

import pytest
from fastapi.testclient import TestClient

from src.contracts import Intent
from src.control.pulse_map import center
from src.io.settings import load
from src.loop import Loop
from src.web.server import make_app

pytestmark = pytest.mark.sim

DT = 0.01


class SteadyHand:
    """A hand process that never drops out, always tilting the same way."""

    seen, calibrating, progress = True, False, 1.0

    def __init__(self, clock):
        self.clock = clock

    def latest(self):
        return Intent(0.8, 0.0, True, self.clock[0])

    def recentre(self):
        pass


@pytest.fixture
def clock():
    return [1000.0]


@pytest.fixture
def loop(clock):
    return Loop(load(), SteadyHand(clock))


def run(loop, clock, n, page):
    for _ in range(n):
        clock[0] += DT
        if page:
            loop.page_seen(clock[0])
        loop.tick(clock[0], DT)


def off_level(loop):
    return abs(loop.last.servo_cmd.x_us - center(loop.cfg["rig"])[0])


def test_a_quiet_page_ramps_the_hand_to_level_and_a_returning_page_steers_again(loop, clock):
    rig, quiet_s = loop.cfg["rig"], loop.cfg["game"]["heartbeat_stale_s"]
    loop.arm()
    run(loop, clock, 100, page=True)
    steered = off_level(loop)
    assert steered > 50 and not loop.stale and not loop.snapshot()["page_stale"]

    run(loop, clock, round(quiet_s / DT) - 1, page=False)
    assert not loop.stale and off_level(loop) == steered

    left = []
    for _ in range(100):
        run(loop, clock, 1, page=False)
        left.append(off_level(loop))
    steps = [a - b for a, b in pairwise([steered, *left])]
    assert all(s >= 0 for s in steps) and max(steps) <= rig["kill_ramp_us_s"] * DT + 1
    assert left[-1] == 0 and loop.safety.armed and loop.last.armed
    assert loop.stale and loop.snapshot()["page_stale"]

    run(loop, clock, 100, page=True)
    assert off_level(loop) == steered and not loop.stale and not loop.page_stale


def test_a_loop_no_page_ever_talked_to_follows_the_hand(loop, clock):
    loop.arm()
    run(loop, clock, 200, page=False)
    assert off_level(loop) > 50 and not loop.stale and not loop.page_stale


def test_only_a_controlling_page_counts_and_arming_from_it_starts_the_watch(loop):
    with TestClient(make_app(loop)).websocket_connect("/ws") as ws:
        ws.send_json({})
        assert ws.receive_json()["page_stale"] is False
        assert loop.page_t is None
        ws.send_json({"x": 0, "y": 0, "arm": True})
        ws.receive_json()
        seen = loop.page_t
        assert loop.safety.armed and seen is not None
        ws.send_json({})
        ws.receive_json()
    assert loop.page_t == seen
