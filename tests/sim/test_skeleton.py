import time

import pytest
from fastapi.testclient import TestClient

from src.app import Loop
from src.io.settings import load
from src.web.server import make_app

pytestmark = pytest.mark.sim


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def ticks(loop, clock, n, key=None, dt=0.01):
    for _ in range(n):
        clock[0] += dt
        if key:
            loop.keys.set(*key)
        loop.tick(clock[0], dt)


def test_key_tilts_sim_plate_through_safety_then_kill_levels_and_disarms(clock):
    loop = Loop(load())
    rig = loop.cfg["rig"]
    loop.arm()

    ticks(loop, clock, 100, key=(1.0, 0.0))
    p = loop.last.plate_state
    assert p.sent_x_deg == pytest.approx(rig["startup_range_us"] / rig["us_per_deg_x"], abs=0.05)
    assert p.sent_y_deg == 0
    assert loop.last.armed

    loop.kill()
    ticks(loop, clock, 300, key=(1.0, 0.0))
    assert not loop.last.armed
    assert loop.last.plate_state.sent_x_deg == 0


def test_no_input_keeps_plate_level(clock):
    loop = Loop(load())
    loop.arm()
    ticks(loop, clock, 100)
    assert loop.last.plate_state.sent_x_deg == 0
    assert loop.last.plate_state.sent_y_deg == 0


def test_browser_page_and_socket():
    loop = Loop(load())
    client = TestClient(make_app(loop))
    assert "On-Tilt" in client.get("/").text
    with client.websocket_connect("/ws") as sock:
        sock.send_json({"x": 0.5, "y": 0.0, "arm": True})
        sock.receive_json()
    assert loop.safety.armed
    assert loop.keys.latest().x == 0.5


def test_sim_ball_starts_centred_and_rolls_downhill(clock):
    loop = Loop(load())
    loop.arm()
    ticks(loop, clock, 1)
    assert (loop.ball.x_mm, loop.ball.y_mm) == (0, 0) and loop.ball.valid

    ticks(loop, clock, 100, key=(1.0, 0.0))
    assert loop.ball.x_mm > 0 and loop.ball.y_mm == 0
    assert loop.last.ball.x_mm > 0
