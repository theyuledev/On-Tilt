import re
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.control.pulse_map import center
from src.inputs import keyboard
from src.io.settings import load
from src.loop import Loop
from src.safety import LEVEL_HOLD_S, limits
from src.web.server import STATIC, make_app

pytestmark = pytest.mark.sim

DT = 0.01
RIGHT = {"x": 1, "y": 0}
# snapshot fields the page reads that the bare loop does not send
NOT_SENT = {
    "race": "only there once a game is wired",
    "guard": "not sent yet, the board skips the glow without it",
}
FN = re.compile(r"^function \w+\(([^)]*)\) \{$(.*?)^\}", re.M | re.S)
# ways a reconnect can drop a queued arm; add forms here, don't loosen these
DROPS_ARM = re.compile(
    r"delete pending\.arm\b"
    r"|pending\.arm\s*=\s*(?:false|undefined|null)\b"
    r"|(?<![\w.])pending\s*=\s*\{"
)
ARM_DISABLED = r"""\$\(\s*["']arm["']\s*\)\.disabled\s*="""


@pytest.fixture
def page(monkeypatch):
    now = [1000.0]
    # only the key stamps follow the test clock, the socket keeps real time
    monkeypatch.setattr(keyboard, "time", SimpleNamespace(monotonic=lambda: now[0]))
    loop = Loop(load())
    client = TestClient(make_app(loop))
    with client.websocket_connect("/ws") as sock:
        yield SimpleNamespace(loop=loop, client=client, sock=sock, now=now)


def say(p, msg):
    p.sock.send_json(msg)
    return p.sock.receive_json()


def run(p, n, msg=None):
    # the page sends its held keys once per frame, one frame per tick here
    recs = []
    for _ in range(n):
        p.now[0] += DT
        if msg is not None:
            say(p, msg)
        recs.append(p.loop.tick(p.now[0], DT))
    return recs


def until(p, done, msg=None, cap=1000):
    recs = []
    while not done() and len(recs) < cap:
        recs += run(p, 1, msg)
    return recs


def assert_ramp(rig, recs):
    xs = [r.servo_cmd.x_us for r in recs if r.armed]
    steps = [b - a for a, b in pairwise(xs)]
    assert all(-rig["kill_ramp_us_s"] * DT - 1 <= d <= 0 for d in steps)


def test_arm_and_keys_over_ws_tilt_the_sim_plate_and_roll_the_ball(page):
    say(page, {"arm": True})
    assert page.loop.safety.armed
    p = run(page, 100, RIGHT)[-1].plate_state
    assert p.sent_x_deg > 0 and p.sent_y_deg == 0
    assert page.loop.ball.x_mm > 0 and page.loop.ball.y_mm == 0
    assert say(page, RIGHT)["armed"]


def test_kill_over_ws_ramps_level_then_disarms_and_only_then_rearms(page):
    rig = page.loop.cfg["rig"]
    cx, cy = center(rig)
    say(page, {"arm": True})
    tilted = run(page, 100, RIGHT)[-1]
    assert cx < tilted.servo_cmd.x_us <= limits(rig)[1]

    say(page, {"kill": True, **RIGHT})
    ramp = run(page, 10, RIGHT)
    held = replace(page.loop.safety)
    for msg in ({"kill": True, **RIGHT}, {"arm": True, **RIGHT}):
        assert say(page, msg)["killing"]
        assert page.loop.safety == held
    ramp += until(page, lambda: not page.loop.safety.armed, RIGHT)

    on = [tilted] + [r for r in ramp if r.armed]
    assert_ramp(rig, on)
    assert all(r.servo_cmd.y_us == cy for r in on)
    level = [r for r in on if (r.servo_cmd.x_us, r.servo_cmd.y_us) == (cx, cy)]
    assert on[-1] is level[-1] and not run(page, 1, RIGHT)[-1].armed
    # the hold counts from the first level pulse sent
    assert (len(level) - 1) * DT == pytest.approx(LEVEL_HOLD_S, abs=DT)

    say(page, {"arm": True, **RIGHT})
    assert run(page, 1, RIGHT)[-1].armed and say(page, RIGHT)["armed"]


def test_arm_with_kill_stays_disarmed_and_a_watch_tab_leaves_the_keys(page):
    say(page, {"arm": True, "kill": True})
    run(page, 1)
    assert not page.loop.safety.armed and not say(page, {})["armed"]

    say(page, {"x": 0.5, "y": 0})
    keys = page.loop.keys.latest()
    with page.client.websocket_connect("/ws") as watch:
        for _ in range(3):
            page.now[0] += DT
            watch.send_json({})
            watch.receive_json()
    assert page.loop.keys.latest() == keys


def test_quiet_page_goes_stale_and_the_plate_ramps_level(page):
    rig = page.loop.cfg["rig"]
    cx, _ = center(rig)
    say(page, {"arm": True})
    tilted = run(page, 100, RIGHT)[-1]
    t_key = page.loop.keys.latest().t_s

    quiet = until(page, lambda: page.loop.stale)
    assert quiet[-1].t_s - t_key > rig["stale_intent_s"] >= quiet[-2].t_s - t_key
    ramp = until(page, lambda: page.loop.last.servo_cmd.x_us == cx)
    assert_ramp(rig, [tilted] + quiet + ramp)

    snap = say(page, {})
    assert snap["stale"] and snap["armed"] and snap["plate_state"]["sent_x_deg"] == 0


def page_reads():
    reads = set()
    for js in STATIC.glob("*.js"):
        for params, body in FN.findall(js.read_text()):
            if "s" not in [a.strip() for a in params.split(",")]:
                continue
            reads |= set(re.findall(r"(?<![\w.])s\.(\w+(?:\.\w+)?)", body))
            # const b = s.ball, p = s.plate_state; then b.valid, p.sent_x_deg
            for alias, field in re.findall(r"(\w+) = s\.(\w+)[,;]", body):
                subs = re.findall(rf"(?<![\w.]){alias}\.(\w+)", body)
                reads |= {f"{field}.{f}" for f in subs}
    return reads


def test_every_snapshot_field_the_page_reads_is_sent(page):
    run(page, 1)
    snap = say(page, {})
    reads = page_reads()
    seen = {"armed", "t_s", "intent", "ball.x_mm", "ball.vx_mm_s", "plate_state.sent_y_deg"}
    assert seen <= reads
    assert not NOT_SENT.keys() & snap.keys(), "sent now, drop it from NOT_SENT"
    for read in reads | {"intent.x", "intent.y"}:  # padDot gets s.intent
        top, _, sub = read.partition(".")
        if top in NOT_SENT:
            continue
        assert top in snap, read
        assert not sub or sub in snap[top], read


def test_beat_pill_t_s_moves_with_ticks_not_with_messages(page):
    beat = page.client.get("/config").json()["heartbeat_stale_s"]
    assert beat == load()["game"]["heartbeat_stale_s"]
    run(page, 1)
    seen = [say(page, {})["t_s"] for _ in range(3)]
    assert len(set(seen)) == 1
    for _ in range(3):
        run(page, 1)
        seen.append(say(page, {})["t_s"])
        assert seen[-1] > seen[-2]


def handler(name):
    js = (STATIC / "app.js").read_text()
    start = js.index("=>", js.index(f"ws.{name} =")) + 2
    depth = 0
    for k in range(start, len(js)):
        depth += (js[k] in "({[") - (js[k] in ")}]")
        if depth == 0 and js[k] in ";\n":
            return js[start:k]
    return js[start:]


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="an arm clicked while offline goes out on reconnect"
)
def test_an_arm_clicked_while_offline_is_not_sent_on_reconnect():
    assert DROPS_ARM.search(handler("onclose"))


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="arm stays clickable while the link is down"
)
def test_arm_is_disabled_while_the_link_is_down():
    assert re.search(ARM_DISABLED + r"\s*true\b", handler("onclose"))
    assert re.search(ARM_DISABLED + r"(?!=)", handler("onopen"))
