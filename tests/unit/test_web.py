import re
import time
from functools import partial
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.games import maze
from src.inputs.keyboard import KeyboardInput
from src.io.ports import Frame
from src.io.settings import load
from src.safety import SafetyState
from src.web.server import JPEG_PERIOD_S, STATIC, make_app

pytestmark = pytest.mark.unit

HUD_IDS = ("race", "timer", "cps", "banner", "kill", "seen", "pad", "level", "ballx", "tilt")
LIGHT_IDS = ("link", "board", "camlight", "ballpill", "hand", "beat", "stale")
RACE_IDS = ("racebtn", "count", "split", "delta", "best", "pen", "result")
RACE_FIELDS = ("state", "elapsed_s", "reached", "delta_s", "penalty_s", "reason", "ghost_lap_s")
RESULT_IDS = ("respen", "resname", "ressave", "ressaved", "resarm")


@pytest.fixture
def client():
    return TestClient(make_app(SimpleNamespace(cfg=load(), hand=None)))


def static(client, name):
    r = client.get(f"/static/{name}")
    assert r.status_code == 200
    return r.text


def page_at(client, text):
    return client.get("/").text.index(text)


def test_page_has_the_race_hud_lights_and_kill(client):
    page = client.get("/").text
    assert all(f'id="{i}"' in page for i in HUD_IDS + LIGHT_IDS)
    assert "00:00.00" in page and "NO INPUT" in page and 'id="side"' in page
    assert "/static/style.css" in page and "/static/app.js" in page and "/static/race.js" in page
    assert "/static/pad.js" in page


def test_both_fonts_are_served_from_static_and_nothing_loads_from_the_network(client):
    css = static(client, "style.css")
    for name, family in (("PressStart2P", "Press Start 2P"), ("VT323", "VT323")):
        assert f'@font-face {{ font-family: "{family}"; src: url("fonts/{name}.woff2")' in css
        font = client.get(f"/static/fonts/{name}.woff2")
        assert font.status_code == 200 and font.content[:4] == b"wOF2"
    for path in STATIC.iterdir():
        if path.suffix in (".html", ".css", ".js"):
            text = path.read_text()
            assert "http://" not in text and "https://" not in text and "//fonts" not in text, path


def test_page_has_the_view_tag_the_board_map_and_loads_the_camera_before_the_views(client):
    page = client.get("/").text
    assert all(f'id="{i}"' in page for i in ("viewtag", "minimap", "mini"))
    order = ("race", "gates", "cam", "plate3d", "follow", "view", "choices", "app")
    at = [page.index(f"/static/{f}.js") for f in order]
    assert at == sorted(at)
    assert "drawView(ctx, cfg, maze, s);" in static(client, "app.js")
    lean = "transform-origin: 50% 58%; transform: perspective(1000px) rotateX(34deg) scale(1.32);"
    css = static(client, "style.css")
    assert f"#race.follow {{ {lean} }}" in css
    # the map boxes what that scale leaves in view
    assert "const FOLLOW_SCALE = 1.32;" in static(client, "follow.js")
    # the plain top-down board has no tag on it
    assert 'top-down" hidden></button>' in page
    assert "#viewtag { position: absolute; left: 22px; bottom: 20px;" in css


def test_pops_are_sized_off_the_board_panel_wrap_and_stack_clear_of_each_other(client):
    css = static(client, "style.css")
    assert "container-type: inline-size;" in css.split(".arena {")[1].split("}")[0]
    hud = css.split("\n.hud {")[1].split("}")[0]
    assert "font: 400 24px var(--px); font-size: clamp(12px, 4cqw, 24px);" in hud
    assert "max-width: 90%;" in hud and "white-space: pre-wrap;" in hud
    order = ("split", "pen", "result", "count", "level")
    tops = [int(re.search(rf"^#{i} {{ top: (\d+)%;", css, re.M)[1]) for i in order]
    assert tops == [14, 24, 34, 50, 72]
    assert "#side:has(#balance:not([hidden])) { gap: 16px; }" in css


def test_page_scripts_never_define_the_same_name_twice(client):
    names = []
    for src in re.findall(r'<script src="/static/([\w.]+)"', client.get("/").text):
        names += re.findall(r"^(?:function|const|let) (\w+)", static(client, src), re.M)
    assert len(names) > 100 and sorted(names) == sorted(set(names))


def test_play_and_settings_pages_link_to_each_other(client):
    assert '<a id="settings" class="pill" href="/settings">SETTINGS</a>' in client.get("/").text
    page = client.get("/settings").text
    assert '<a class="pill" href="/">BACK TO PLAY</a>' in page and "/static/settings.js" in page
    assert ".pill.off { opacity: 0.4; pointer-events: none; }" in static(client, "style.css")


def test_choice_controls_start_disabled_until_the_first_snapshot(client):
    page = client.get("/").text
    assert 'id="choices" class="locked"' in page and page.count("data-pick=") == 10
    for at in [m.start() for m in re.finditer("data-pick=", page)] + [page.index('id="guard"')]:
        tag = page[page.rindex("<button", 0, at) : page.index(">", at)]
        assert tag.endswith(" disabled")


def test_page_has_the_race_hud_and_loads_it_between_race_and_app(client):
    page = client.get("/").text
    assert all(f'id="{i}"' in page for i in RACE_IDS)
    order = ("race.js", "hud.js", "results.js", "guide.js", "app.js")
    at = [page.index(f"/static/{f}") for f in order]
    assert at == sorted(at)


def test_banner_line_comes_from_guide_and_starts_on_connecting(client):
    assert "function guide(s, ctx)" in static(client, "guide.js")
    assert 'data-tone="warn">connecting...</span>' in client.get("/").text
    js = static(client, "app.js")
    assert "guide(s, { live, watch, hand: cfg.hand, lost, gates })" in js
    assert 'light("link", "OFFLINE", "warn"); banner(null, false, false);' in js
    css = static(client, "style.css")
    assert '#banner[data-tone="stop"] { color: var(--red); }' in css
    # the line sits under the board so it never covers the ball
    assert page_at(client, 'class="arena"') < page_at(client, 'id="banner"')


def test_hud_is_keyed_on_the_race_view(client):
    js = static(client, "hud.js")
    assert "function hud(s, maze)" in js and "hud(s, maze)" in static(client, "app.js")
    assert all(f"r.{f}" in js for f in RACE_FIELDS)
    assert "s.race && s.race.state" in js and "r.off_path === true" in js
    assert "#3EE6E0" in js and "#E69F00" in js and "FIRST LAP" in js
    assert "fmtLap(" in js and "maze.checkpoints.length" in js
    assert '"false start"' in js and "arm, then R" in js


def test_race_js_lights_reached_gates_and_pulses_the_next(client):
    js = static(client, "gates.js")
    assert "function gates(ctx, maze, ballR, race)" in js
    assert "race.reached" in js and "GATE_PULSE_MS" in js and "GATE_FLASH_MS" in js
    assert "gates(ctx, maze, r, s.race)" in static(client, "race.js")


def test_r_key_and_race_button_send_a_one_shot_start_or_abort(client):
    js = static(client, "app.js")
    assert '"KeyR"' in js and "pending.race = raceCmd()" in js
    assert all(g in js for g in ("e.repeat", "e.metaKey", "e.ctrlKey", "e.altKey"))
    assert '"INPUT"' in js and '"TEXTAREA"' in js
    hud = static(client, "hud.js")
    assert '"abort" : "start"' in hud
    assert 'PRE = ["calibrating", "staging"]' in hud and '[...PRE, "countdown", "running"]' in hud


def test_watch_tab_disables_the_race_button(client):
    assert '$("racebtn").disabled = true' in static(client, "app.js")


def test_ws_race_start_and_abort_reach_the_game():
    calls = []
    game = SimpleNamespace(start=lambda: calls.append("start"), abort=lambda: calls.append("abort"))
    loop = SimpleNamespace(cfg=load(), game=game, snapshot=lambda: {})
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        for msg in ({"race": "start"}, {}, {"race": "abort"}, {"race": "nope"}):
            sock.send_json(msg)
            sock.receive_json()
    assert calls == ["start", "abort"]


def test_ws_race_key_without_a_game_keeps_the_socket_open():
    calls = []
    loop = SimpleNamespace(
        cfg=load(),
        keys=KeyboardInput({"key_ramp_s": 0.0}),
        arm=lambda: calls.append("arm"),
        kill=lambda: calls.append("kill"),
        snapshot=lambda: {},
    )
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        sock.send_json({"race": "start"})
        sock.receive_json()
        sock.send_json({"kill": True})
        sock.receive_json()
    assert calls == ["kill"]


def test_ws_save_hands_the_game_a_cleaned_name():
    saved = []
    loop = SimpleNamespace(cfg=load(), game=SimpleNamespace(save=saved.append), snapshot=lambda: {})
    names = ("../../x y!", "?/.\\*", "   ", "averyverylongname", "abcdefg hij", 42)
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        for name in names:
            sock.send_json({"save": name})
            sock.receive_json()
    assert saved == ["x y", "averyver", "abcdefg", "42"]


def test_ws_save_without_a_game_is_ignored_and_the_socket_stays_open():
    loop = SimpleNamespace(cfg=load(), snapshot=lambda: {"ok": 1})
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        for msg in ({"save": "x"}, {}):
            sock.send_json(msg)
            assert sock.receive_json() == {"ok": 1}


def _ws_loop(calls):
    keys = SimpleNamespace(set=lambda x, y: calls.append("keys"))
    return SimpleNamespace(
        cfg=load(),
        keys=keys,
        arm=lambda: calls.append("arm"),
        kill=lambda: calls.append("kill"),
        snapshot=lambda: {},
    )


def test_ws_bad_x_drops_only_that_message():
    calls = []
    with TestClient(make_app(_ws_loop(calls))).websocket_connect("/ws") as sock:
        sock.send_json({"x": "abc", "y": 0})
        sock.receive_json()
        sock.send_json({"kill": True})
        sock.receive_json()
    assert calls == ["kill"]


def test_ws_kill_with_a_bad_x_still_kills():
    calls = []
    with TestClient(make_app(_ws_loop(calls))).websocket_connect("/ws") as sock:
        sock.send_json({"kill": True, "x": "abc", "y": 0})
        sock.receive_json()
    assert calls == ["kill"]


def test_ws_non_dict_message_keeps_the_socket_open():
    calls = []
    with TestClient(make_app(_ws_loop(calls))).websocket_connect("/ws") as sock:
        sock.send_json([1, 2])
        sock.receive_json()
        sock.send_json({"kill": True})
        sock.receive_json()
    assert calls == ["kill"]


def test_ws_save_that_raises_keeps_the_socket_open():
    def save(name):
        raise OSError("disk full")

    calls = []
    loop = _ws_loop(calls)
    loop.game = SimpleNamespace(save=save)
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        sock.send_json({"save": "abc"})
        sock.receive_json()
        sock.send_json({"kill": True})
        sock.receive_json()
    assert calls == ["kill"]


def _safety_loop():
    st = SafetyState()
    return st, SimpleNamespace(cfg=load(), arm=st.arm, kill=st.kill, snapshot=lambda: {})


def test_ws_arm_and_kill_in_one_message_from_disarmed_stays_disarmed():
    st, loop = _safety_loop()
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        sock.send_json({"arm": True, "kill": True})
        sock.receive_json()
    assert not st.armed


def test_ws_arm_alone_from_disarmed_arms():
    st, loop = _safety_loop()
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        sock.send_json({"arm": True})
        sock.receive_json()
    assert st.armed and not st.killing


def test_page_has_a_results_panel_loaded_after_hud_and_before_app(client):
    page = client.get("/").text
    assert all(f'id="{i}"' in page for i in RESULT_IDS)
    assert page.index('id="side"') < page.index('id="results" class="field" hidden')
    assert '<input id="resname" maxlength="8"' in page
    at = [page.index(f"/static/{f}.js") for f in ("hud", "results", "app")]
    assert at == sorted(at)
    assert "hud(s, maze);\n  results(s);" in static(client, "app.js")


def test_typing_in_a_field_skips_the_window_keys_but_still_lets_go_of_tilt(client):
    js = static(client, "app.js")
    down = js[js.index('addEventListener("keydown"') : js.index('addEventListener("keyup"')]
    assert down.index("if (typing(e)) return;") < down.index("clickKeys.has(e.code)")
    up = js[js.index('addEventListener("keyup"') :].split("\n")[0]
    assert up.index("!typing(e)") < up.index("clickKeys.has(e.code)")
    assert up.endswith("e.preventDefault(); held.delete(e.code); });")


def test_space_kills_while_armed_even_from_the_name_field(client):
    js = static(client, "app.js")
    down = js[js.index('addEventListener("keydown"') : js.index('addEventListener("keyup"')]
    kill = 'if (e.code === "Space" && armedNow) { pending.kill = true; e.preventDefault(); '
    assert down.index(kill + "return; }") < down.index("if (typing(e)) return;")
    show = js[js.index("function show(s) {") :]
    assert show.index("armedNow = !!s.armed;") < show.index("\n}")


def test_page_has_a_hand_pill_keyed_on_the_snapshot_flags(client):
    assert 'id="hand"' in client.get("/").text
    js = static(client, "app.js")
    assert "HOLD STILL" in js and "s.hand_seen" in js and "s.hand_calibrating" in js


def test_pad_draws_the_intent_dot_and_guards_the_progress_ring(client):
    js = static(client, "pad.js")
    assert "function drawPad(ctx, cfg, s)" in js
    assert "cfg.dead_zone" in js and "s.intent" in js
    assert "s.hand_progress !== undefined" in js
    # pad.js leans on these race.js helpers
    race = static(client, "race.js")
    assert all(f"function {f}(" in race for f in ("fit", "circle", "polyline"))
    assert "drawPad(padCtx, cfg, s)" in static(client, "app.js")


def test_hand_lost_banner_needs_a_hand_seen_since_arming(client):
    js = static(client, "app.js")
    assert '"HAND LOST"' in static(client, "guide.js") and "if (!s.armed) handWasSeen = false" in js
    assert "s.armed && handWasSeen && !s.hand_seen" in js
    assert '$("level").hidden = !(s.stale && s.armed && !s.hand_calibrating)' in js


def test_page_tracks_held_keys_by_code_so_shift_cannot_stick_one(client):
    js = static(client, "app.js")
    assert "e.code" in js and "e.key" not in js
    assert all(k in js for k in ("KeyA", "KeyD", "KeyW", "KeyS"))
    assert 'addEventListener("blur", () => held.clear())' in js


def test_c_key_sends_a_one_shot_recentre_like_arm_and_kill(client):
    assert "c re-centres the hand neutral" in client.get("/").text
    js = static(client, "app.js")
    assert 'e.code === "KeyC" && !e.metaKey && !e.ctrlKey' in js and "pending.recentre = true" in js


def test_space_or_enter_never_clicks_a_focused_arm_button(client):
    js = static(client, "app.js")
    assert 'clickKeys = new Set(["Space", "Enter", "NumpadEnter"])' in js
    # space clicks a focused button on keyup, so both key events swallow it
    assert js.count("clickKeys.has(e.code)) e.preventDefault();") == 2
    assert 'addEventListener("keyup", e => { if (!typing(e) && clickKeys.has(e.code))' in js
    assert 'if (e.code === "Space") { pending.kill = true; return; }' in js
    for b in ("arm", "kill"):
        assert f'$("{b}").onclick = e => {{ pending.{b} = true; e.currentTarget.blur(); }}' in js
    assert '$("racebtn").onclick = e => { pending.race = raceCmd(); e.currentTarget.blur(); }' in js


def test_watch_tab_sends_nothing_and_takes_no_input(client):
    js = static(client, "app.js")
    assert 'has("watch")' in js and "watch ? {} :" in js


class Recentres:
    def __init__(self):
        self.n = 0

    def recentre(self):
        self.n += 1


def test_ws_recentre_reaches_the_hand_and_is_harmless_without_one():
    hand = Recentres()
    loop = SimpleNamespace(cfg=load(), hand=hand, snapshot=lambda: {"ok": 1})
    with TestClient(make_app(loop)).websocket_connect("/ws") as ws:
        ws.send_json({"recentre": True})
        assert ws.receive_json() == {"ok": 1}
        ws.send_json({})
        ws.receive_json()
    assert hand.n == 1
    loop.hand = None
    with TestClient(make_app(loop)).websocket_connect("/ws") as ws:
        ws.send_json({"recentre": True})
        assert ws.receive_json() == {"ok": 1}


def test_config_gives_plate_ball_and_stale_limit_from_settings(client):
    cfg = load()
    assert client.get("/config").json() == {
        "plate_inner_mm": cfg["rig"]["plate_inner_mm"],
        "ball_d_mm": cfg["rig"]["ball_d_mm"],
        "camera_width": cfg["rig"]["camera_width"],
        "heartbeat_stale_s": cfg["game"]["heartbeat_stale_s"],
        "dead_zone": cfg["input"]["dead_zone"],
        "hand": False,
        "tilt_full_deg": cfg["input"]["intent_gain_deg"],
        "px_per_mm": cfg["vision"]["px_per_mm"],
        "camera_fps": cfg["rig"]["camera_fps"],
        "checkpoint_r_mm": cfg["game"]["checkpoint_r_mm"],
    }
    with_hand = TestClient(make_app(SimpleNamespace(cfg=cfg, hand=object())))
    assert with_hand.get("/config").json()["hand"] is True


def race_maze(name):
    cfg = load()
    cfg["game"]["maze"] = name
    return TestClient(make_app(SimpleNamespace(cfg=cfg))).get("/race/maze")


def test_race_maze_is_the_configured_maze_as_lists(client):
    m = maze.load(load()["game"]["maze"])
    assert client.get("/race/maze").json() == {
        "points": [list(p) for p in m.points],
        "checkpoints": list(m.checkpoints),
        "start": list(m.start),
        "walls": [list(map(list, w)) for w in m.walls],
    }


def test_race_maze_sends_the_walls_and_none_for_the_square(monkeypatch, tmp_path):
    assert race_maze("square").json()["walls"] == []
    (tmp_path / "w.toml").write_text(
        "points = [[0, 0], [10, 0]]\ncheckpoints = [1]\nstart = [0, 0]\n"
        "walls = [[[-20, 5], [20, 5]], [[-20, -5], [0, -5], [20, -5]]]\n"
    )
    monkeypatch.setattr(maze, "load", partial(maze.load, root=tmp_path))
    assert race_maze("w").json()["walls"] == [[[-20, 5], [20, 5]], [[-20, -5], [0, -5], [20, -5]]]


@pytest.mark.parametrize(
    "body",
    [
        None,
        "not toml [",
        "checkpoints = [0]\nstart = [0, 0]\n",
        "points = [[0, 0]]\ncheckpoints = [3]\nstart = [0, 0]\n",
        "points = 5\ncheckpoints = [0]\nstart = [0, 0]\n",
    ],
)
def test_race_maze_is_404_for_a_missing_or_bad_maze(monkeypatch, tmp_path, body):
    if body is not None:
        (tmp_path / "bad.toml").write_text(body)
    monkeypatch.setattr(maze, "load", partial(maze.load, root=tmp_path))
    assert race_maze("bad").status_code == 404


def test_page_still_connects_and_draws_the_plate_without_a_maze(client):
    app = static(client, "app.js")
    assert 'json("/race/maze").catch(() => null)]).then(' in app
    assert "if (maze) lamps(maze.checkpoints.length" in static(client, "hud.js")
    js = static(client, "race.js")
    assert "if (maze) {\n    if (maze.walls) (look.tall ? tallWalls : walls)(ctx, maze);" in js
    assert js.index("if (maze) {") < js.index("gates(ctx, maze, r, s.race)")


def test_race_js_draws_walls_under_the_race_line_only_when_the_maze_has_them(client):
    js = static(client, "race.js")
    assert "function walls(ctx, maze, color = MARKER, width = 3.2)" in js
    assert "maze.walls.forEach((w) => polyline(ctx, w))" in js
    guarded = js.index("if (maze.walls) (look.tall ? tallWalls : walls)(ctx, maze);")
    # the S and the E are marker too: over the walls, under the race line
    assert guarded < js.index("    ends(ctx, maze, h, r);") < js.index("    route(ctx, maze);")


def test_race_js_draws_the_plate_in_mm_with_a_ball_trail(client):
    js = static(client, "race.js")
    assert "function draw(ctx, cfg, maze, s, look)" in js
    assert "cfg.plate_inner_mm" in js and "TRAIL_S" in js and "s.ball.valid" in js


def test_race_js_draws_the_ghost_under_the_player(client):
    js = static(client, "race.js")
    assert "s.race" in js and "ghost_mm" in js and "#FF5FA2" in js
    assert js.index("if (g) ghost(ctx, g, r, s.t_s)") < js.index("  ball(ctx, at, r, h);")


def test_page_has_a_strip_loaded_after_race_and_before_app(client):
    page = client.get("/").text
    assert 'id="strip"' in page
    at = [page.index(f"/static/{name}.js") for name in ("race", "pad", "strip", "app")]
    assert at == sorted(at)
    assert "strip(cfg, s);" in static(client, "app.js")


def test_strip_plots_ball_error_and_sent_tilt_over_the_last_5_s(client):
    js = static(client, "strip.js")
    assert "s.ball" in js and "b.valid" in js and "sent_x_deg" in js and "sent_y_deg" in js
    assert "const STRIP_S = 5;" in js and "keep(stripBuf, [err, tilt], s.t_s, STRIP_S)" in js
    assert "now - buf[0].t > keepS" in static(client, "race.js")
    assert "fit(c)" in js and "ERR --" in js


def test_strip_err_has_a_fixed_plate_half_width_scale_and_tilt_autoscales(client):
    js = static(client, "strip.js")
    assert '"x", cfg.plate_inner_mm / 2, ERR_COLOR' in js
    assert "Math.min(p[key] / top, 1)" in js
    assert '"y", Math.max(...stripBuf.map(q => q.y)) || 1, TILT_COLOR' in js


def test_page_rings_the_tracked_ball_on_the_camera_image(client):
    page = client.get("/").text
    assert page.index('id="cam"') < page.index('id="seen"') < page.index('id="nocam"')
    js = static(client, "app.js")
    assert "s.ball_px" in js and "cfg.camera_width" in js


def test_page_says_when_config_fails_and_retries_a_lost_camera(client):
    js = static(client, "app.js")
    assert "r.ok ? r.json()" in js and ".catch(" in js and "load failed" in js
    assert "camRetry = setTimeout(nextCam, 1000)" in js
    # hidden must beat #cam's display: block or a broken image icon shows
    assert "[hidden] { display: none !important; }" in static(client, "style.css")


def test_camera_jpeg_is_half_res_and_cached_for_a_frame_period(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    loop = SimpleNamespace(cfg=load(), camera_lost=False)
    loop.frame = Frame(np.zeros((800, 1280, 3), np.uint8), 0.0)
    client = TestClient(make_app(loop))

    first = client.get("/camera.jpg").content
    img = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (400, 640)

    loop.frame = Frame(np.full((800, 1280, 3), 255, np.uint8), 0.01)
    now[0] += JPEG_PERIOD_S / 2
    assert client.get("/camera.jpg").content == first
    now[0] += JPEG_PERIOD_S
    assert client.get("/camera.jpg").content != first


def test_camera_jpeg_full_is_full_res_and_skips_the_cache(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    loop = SimpleNamespace(cfg=load(), camera_lost=False)
    loop.frame = Frame(np.zeros((800, 1280, 3), np.uint8), 0.0)
    client = TestClient(make_app(loop))

    small = client.get("/camera.jpg").content
    full = client.get("/camera.jpg?full=1").content
    assert full != small
    img = cv2.imdecode(np.frombuffer(full, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (800, 1280)

    loop.frame = Frame(np.full((800, 1280, 3), 255, np.uint8), 0.01)
    assert client.get("/camera.jpg?full=1").content != full
    assert client.get("/camera.jpg").content == small


def test_camera_jpeg_404s_without_a_frame():
    loop = SimpleNamespace(cfg=load(), camera_lost=False, frame=None)
    assert TestClient(make_app(loop)).get("/camera.jpg").status_code == 404


def test_ws_ignores_keys_that_are_not_finite():
    keys = KeyboardInput({"key_ramp_s": 0.0})
    loop = SimpleNamespace(cfg=load(), keys=keys, snapshot=lambda: {})
    with TestClient(make_app(loop)).websocket_connect("/ws") as ws:
        for msg in ({"x": "nan", "y": 0}, {"x": 0.5, "y": "inf"}):
            ws.send_json(msg)
            ws.receive_json()
        assert not keys.latest().valid
        ws.send_json({"x": 0.5, "y": -0.5})
        ws.receive_json()
    assert keys.latest().valid and (keys.latest().x, keys.latest().y) == (0.5, -0.5)
