import math
import re
import time

import pytest
from fastapi.testclient import TestClient

from src.app import Loop
from src.contracts import BallState, PlateCommand
from src.control.pulse_map import center, tilt_to_pulse
from src.io.serial_link import BoardLink, SerialActuator
from src.io.settings import load
from src.modes import fade_in, hand
from src.web.server import make_app
from tests.contract.fake_board import FakeBoard

pytestmark = pytest.mark.sim

DT = 0.01
TRIM_US = -80
FULL = (1.0, 1.0)
HALF = (0.5, 0.0)  # inside the startup range, so the faded target is never clamped
ONE_SHOTS = ("arm", "hold", "race", "recentre")


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
    loop.arm()
    loop.hold()
    ticks(loop, clock, 20, FULL)
    assert pulse(loop) == level(loop)
    loop.arm()
    assert not loop.holding and not loop.snapshot()["holding"]
    ticks(loop, clock, 100, FULL)
    assert pulse(loop)[0] > level(loop)[0] and loop.safety.armed


def test_hold_while_disarmed_or_board_lost_does_nothing(loop, clock):
    board = FakeBoard()
    idle = Loop(loop.cfg, actuator=SerialActuator(BoardLink(board), loop.cfg["rig"]))
    idle.hold()
    assert not idle.safety.armed and not idle.holding
    ticks(idle, clock, 5, FULL)
    assert board.sent == [] and not board.armed and not idle.snapshot()["holding"]

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
    loop.arm()
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
        loop.arm()
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
        for msg in ({"hold": True, "kill": True}, {"hold": True}):
            sock.send_json(msg)
            sock.receive_json()
        assert not loop.safety.armed and not loop.holding  # hold never arms
        sock.send_json({"arm": True})
        sock.receive_json()
        sock.send_json({"hold": True})
        assert sock.receive_json()["holding"] and loop.safety.armed
        sock.send_json({"hold": True, "kill": True})
        sock.receive_json()
    assert loop.safety.killing and not loop.holding


def test_page_has_a_hold_button_an_h_key_and_a_holding_pill(loop):
    client = TestClient(make_app(loop))
    assert '<button id="hold" disabled>hold (h)</button>' in client.get("/").text
    js = client.get("/static/app.js").text
    assert '"KeyH" && !e.metaKey && !e.ctrlKey) { pending.hold = true; return; }' in js
    assert '$("hold").onclick = e => { pending.hold = true; e.currentTarget.blur(); }' in js
    assert '$("hold").disabled = $("arm").disabled = $("kill").disabled' in js
    assert 's.armed ? (s.holding ? "holding" : "armed")' in js
    shown = js[js.index("function show(s)") : js.index("function connect")]
    lock = '$("hold").disabled = watch || !s.armed || s.killing;'
    assert lock in shown and shown.index(lock) < shown.index("if (!s.plate_state) return;")


def test_a_zero_fade_turns_the_fade_off():
    cmd = PlateCommand(3.0, -2.0)
    for fade_s in (0.0, -1.0):
        assert fade_in(cmd, False, 0.0, {"arm_fade_s": fade_s}) == cmd


def handler(js, name):
    m = re.search(rf"ws\.{name} = \(\) => \{{\n(.*?)\n  \}};", js, re.S)
    assert m, name
    return m.group(1)


def test_offline_drops_a_pending_hold_and_locks_the_button_like_arm(loop):
    js = TestClient(make_app(loop)).get("/static/app.js").text
    close = handler(js, "onclose")
    assert re.search(r"delete pending\.hold;", close)
    assert re.search(r'\$\("hold"\)\.disabled = true;', close)


def test_reconnect_drops_stale_one_shots_before_the_first_send(loop):
    js = TestClient(make_app(loop)).get("/static/app.js").text
    opened = handler(js, "onopen")
    assert re.search(r'\$\("hold"\)\.disabled = watch;', opened)
    dels = [re.search(rf"delete pending\.{k};", opened) for k in ONE_SHOTS]
    assert all(dels) and "pending.kill" not in opened
    assert max(m.start() for m in dels) < opened.index("frame();")
