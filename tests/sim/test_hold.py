import math
import time

import pytest
from fastapi.testclient import TestClient

from src.app import Loop
from src.contracts import BallState
from src.control.pulse_map import center, tilt_to_pulse
from src.io.serial_link import BoardLink, SerialActuator
from src.io.settings import load
from src.modes import hand
from src.web.server import make_app
from tests.contract.fake_board import FakeBoard

pytestmark = pytest.mark.sim

DT = 0.01
TRIM_US = -80
FULL = (1.0, 1.0)
HALF = (0.5, 0.0)  # inside the startup range, so the faded target is never clamped


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def loop():
    cfg = load()
    cfg["rig"] = {**cfg["rig"], "trim_x_us": TRIM_US, "trim_y_us": TRIM_US}
    cfg["input"] = {**cfg["input"], "key_ramp_s": 0.0}
    return Loop(cfg)


def ticks(loop, clock, n, key):
    for _ in range(n):
        clock[0] += DT
        loop.keys.set(*key)
        loop.tick(clock[0], DT)


def pulse(loop):
    return loop.last.servo_cmd.x_us, loop.last.servo_cmd.y_us


def level(loop):
    return tuple(round(v) for v in center(loop.cfg["rig"]))


def slew_ticks(loop):
    rig = loop.cfg["rig"]
    return math.ceil(rig["startup_range_us"] / rig["slew_us_s"] / DT) + 1


def fade_ticks(loop):
    return round(loop.cfg["rig"]["arm_fade_s"] / DT)


def test_hold_settles_on_center_plus_trims_with_keys_still_held(loop, clock):
    loop.arm()
    ticks(loop, clock, 100, FULL)
    assert pulse(loop) != level(loop)
    loop.hold()
    ticks(loop, clock, slew_ticks(loop), FULL)
    for _ in range(200):
        ticks(loop, clock, 1, FULL)
        assert pulse(loop) == level(loop)
    assert loop.safety.armed and loop.last.armed and loop.snapshot()["holding"]


def test_kill_during_hold_ramps_and_disarms_like_any_kill(loop, clock):
    loop.arm()
    ticks(loop, clock, 100, FULL)
    loop.hold()
    ticks(loop, clock, 3, FULL)
    loop.kill()
    ticks(loop, clock, 1, FULL)
    assert loop.safety.killing and not loop.holding
    loop.hold()
    assert not loop.holding  # no hold half way down the ramp
    xs = []
    while loop.last.armed:
        xs.append(pulse(loop)[0])
        ticks(loop, clock, 1, FULL)
    assert len(xs) > 10 and xs == sorted(xs, reverse=True) and xs[-1] == level(loop)[0]
    assert not loop.safety.armed and not loop.holding


def test_arm_during_hold_hands_the_plate_back_to_the_mode(loop, clock):
    loop.hold()
    ticks(loop, clock, 20, FULL)
    assert pulse(loop) == level(loop)
    loop.arm()
    assert not loop.holding and not loop.snapshot()["holding"]
    ticks(loop, clock, 100, FULL)
    assert pulse(loop)[0] > level(loop)[0] and loop.safety.armed


def test_hold_while_disarmed_arms_and_holds_but_not_with_the_board_lost(loop, clock):
    loop.hold()
    assert loop.safety.armed and loop.holding
    ticks(loop, clock, 5, FULL)
    assert loop.last.armed and pulse(loop) == level(loop)

    lost = Loop(loop.cfg)
    lost.board_lost = True
    lost.hold()
    assert not lost.safety.armed and not lost.holding


def test_hold_keeps_the_dead_band_inverter_off_in_a_closed_loop_mode(loop, clock):
    loop.cfg["control"] = {**loop.cfg["control"], "mode": "balance", "inverter_on": True}
    loop.mode = "balance"
    loop.ball = BallState(60.0, -40.0, 0.0, 0.0, valid=True, t_s=clock[0])
    loop.arm()
    ticks(loop, clock, 100, (0.0, 0.0))
    assert pulse(loop) != level(loop)
    loop.hold()
    ticks(loop, clock, slew_ticks(loop), (0.0, 0.0))
    for _ in range(200):
        ticks(loop, clock, 1, (0.0, 0.0))
        assert pulse(loop) == level(loop)


def test_a_refused_arm_keeps_the_hold(loop, clock):
    loop.hold()
    loop.mode, loop.exposure = "balance", "exposure off"
    loop.arm()
    ticks(loop, clock, 20, FULL)
    assert loop.arm_blocked and loop.holding and pulse(loop) == level(loop)


def test_hold_sends_only_the_level_p_once_a_tick(clock):
    board = FakeBoard()
    cfg = load()
    cfg["rig"] = {**cfg["rig"], "trim_x_us": TRIM_US, "trim_y_us": TRIM_US}
    loop = Loop(cfg, actuator=SerialActuator(BoardLink(board), cfg["rig"]))
    loop.arm()
    ticks(loop, clock, 100, FULL)
    loop.hold()
    ticks(loop, clock, slew_ticks(loop), FULL)
    n = len(board.sent)
    ticks(loop, clock, 50, FULL)
    cx, cy = level(loop)
    assert board.sent[n:] == [f"P {cx} {cy}"] * 50 and board.armed

    board.armed = False  # board self levelled and dropped out
    ticks(loop, clock, 2, FULL)
    snap = loop.snapshot()
    assert not snap["armed"] and not snap["holding"]


def faded_in(loop, clock):
    # x offset from level each tick after an arm, with the key already held
    out = []
    for _ in range(fade_ticks(loop) + slew_ticks(loop)):
        ticks(loop, clock, 1, HALF)
        out.append(pulse(loop)[0] - level(loop)[0])
    return out


@pytest.mark.parametrize("before", ["disarmed", "holding"])
def test_tilt_fades_in_from_level_after_an_arm(loop, clock, before):
    ticks(loop, clock, 10, HALF)
    if before == "holding":
        loop.hold()
        ticks(loop, clock, 50, HALF)
    loop.arm()
    offs = faded_in(loop, clock)
    rig = loop.cfg["rig"]
    full = tilt_to_pulse(hand(None, loop.keys.latest(), loop.cfg), rig).x_us - level(loop)[0]
    assert 0 < full < rig["startup_range_us"] and offs[-1] == full
    for i, off in enumerate(offs, 1):
        assert off <= full * min(1.0, i / fade_ticks(loop)) + 1


def test_arm_while_already_armed_does_not_dip_the_plate(loop, clock):
    loop.arm()
    ticks(loop, clock, 100, HALF)
    steady = pulse(loop)
    loop.arm()
    ticks(loop, clock, 1, HALF)
    assert pulse(loop) == steady


def test_ws_hold_holds_and_a_kill_in_the_same_message_wins(loop, clock):
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        sock.send_json({"hold": True, "kill": True})
        sock.receive_json()
        assert not loop.safety.armed and not loop.holding
        sock.send_json({"hold": True})
        assert sock.receive_json()["holding"] and loop.safety.armed
        sock.send_json({"hold": True, "kill": True})
        sock.receive_json()
    assert loop.safety.killing and not loop.holding


def test_page_has_a_hold_button_an_h_key_and_a_holding_pill(loop):
    client = TestClient(make_app(loop))
    assert '<button id="hold">hold (h)</button>' in client.get("/").text
    js = client.get("/static/app.js").text
    assert '"KeyH" && !e.metaKey && !e.ctrlKey) { pending.hold = true; return; }' in js
    assert '$("hold").onclick = e => { pending.hold = true; e.currentTarget.blur(); }' in js
    assert '$("arm").disabled = $("hold").disabled = $("kill").disabled' in js
    assert 's.armed ? (s.holding ? "holding" : "armed")' in js
