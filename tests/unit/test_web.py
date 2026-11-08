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
from src.web.server import JPEG_PERIOD_S, make_app

pytestmark = pytest.mark.unit

HUD_IDS = ("race", "timer", "speed", "cps", "banner", "ballpill", "kill", "seen", "pad", "level")
PILL_IDS = ("hand", "beat", "board", "stale")
RACE_IDS = ("racebtn", "count", "delta", "pen", "result")
RACE_FIELDS = ("state", "elapsed_s", "reached", "delta_s", "penalty_s", "reason", "ghost_lap_s")
RESULT_IDS = ("restime", "respen", "resdelta", "resghost", "resname", "ressaved", "resbeat")


@pytest.fixture
def client():
    return TestClient(make_app(SimpleNamespace(cfg=load(), hand=None)))


def static(client, name):
    r = client.get(f"/static/{name}")
    assert r.status_code == 200
    return r.text


def test_page_has_the_race_hud_pills_and_kill(client):
    page = client.get("/").text
    assert all(f'id="{i}"' in page for i in HUD_IDS + PILL_IDS)
    assert "00:00.00" in page and "no input" in page and 'id="side"' in page
    assert "/static/style.css" in page and "/static/app.js" in page and "/static/race.js" in page
    assert "/static/pad.js" in page


def test_race_hud_hides_without_a_race_and_tilt_reads_big(client):
    page = client.get("/").text
    assert 'id="tilt"' in page and "arrows / wasd to tilt" in page
    assert '$("timer").hidden = $("cps").hidden = s.race === undefined' in static(client, "app.js")
    css = static(client, "style.css")
    assert "#tilt { font: 24px" in css and "color: #fff; white-space: pre" in css


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
    assert "guide(s, { live, watch, hand: cfg.hand, lost })" in js
    assert 'pill("link", "offline", "warn"); banner(null, false, false);' in js
    css = static(client, "style.css")
    assert '#banner[data-tone="stop"] { color: #f33; font-size: 12vmin; }' in css
    # plain hints sit small at the bottom so they never cover the ball
    hint = '#banner[data-tone=""] { place-items: end center; padding-bottom: 12px; '
    assert hint + "font-size: 3vmin; color: #aaa; }" in css


def test_hud_is_keyed_on_the_race_view(client):
    js = static(client, "hud.js")
    assert "function hud(s, maze)" in js and "hud(s, maze)" in static(client, "app.js")
    assert all(f"r.{f}" in js for f in RACE_FIELDS)
    assert "s.race && s.race.state" in js and "r.off_path === true" in js
    assert "#56B4E9" in js and "#E69F00" in js and "FIRST LAP" in js
    assert "fmtLap(" in js and "maze.checkpoints.length" in js
    assert '"false start"' in js and "arm, then R" in js


def test_race_js_lights_reached_gates_and_pulses_the_next(client):
    js = static(client, "race.js")
    assert "function gates(ctx, maze, r, race)" in js and "gates(ctx, maze, r, s.race)" in js
    assert "race.reached" in js and "GATE_PULSE_MS" in js and "GATE_FLASH_MS" in js


def test_r_key_and_race_button_send_a_one_shot_start_or_abort(client):
    js = static(client, "app.js")
    assert '"KeyR"' in js and "pending.race = raceCmd()" in js
    assert all(g in js for g in ("e.repeat", "e.metaKey", "e.ctrlKey", "e.altKey"))
    assert '"INPUT"' in js and '"TEXTAREA"' in js
    hud = static(client, "hud.js")
    assert '"abort" : "start"' in hud
    assert 'PRE = ["calibrating", "staging"]' in hud and '[...PRE, "countdown", "running"]' in hud


def test_hud_shows_no_race_text_while_calibrating_or_staging(client):
    hud = static(client, "hud.js")
    guard = hud.index("if (!r || PRE.includes(r.state))")
    assert 'if (r) $("timer").textContent = $("cps").textContent = ""' in hud
    assert guard < hud.index("countdown(r, was);") < hud.index("split(r, was, performance")
    assert 'toggle("racing", live && !PRE.includes(r.state))' in hud


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
    assert page.index('id="side"') < page.index('id="results" hidden')
    assert '<input id="resname" maxlength="8"' in page
    at = [page.index(f"/static/{f}.js") for f in ("hud", "results", "app")]
    assert at == sorted(at)
    assert "hud(s, maze);\n  results(s);" in static(client, "app.js")


def test_results_panel_shows_the_finished_race_and_saves_a_judge_name(client):
    js = static(client, "results.js")
    assert all(f"r.{f}" in js for f in ("state", "score", "delta_s", "ghost_lap_s"))
    assert 'r.state === "finished" && !!r.score' in js and "RES_DELAY_MS = 1000;" in js
    assert "fmtLap(r.score.time_s)" in js and "signed(d)" in js and "d < 0 ? AHEAD : BEHIND" in js
    assert "JUDGE ${resSaves + 1}" in js and "pending.save = name;" in js
    page = client.get("/").text
    assert ">NEW GHOST<" in page and ">race again: R<" in page and ">beat it<" in page
    assert ">saved<" in page and "#resghost {" in static(client, "style.css")
    assert "#CC79A7" in static(client, "style.css")


def test_name_field_only_takes_keys_while_disarmed(client):
    js = static(client, "results.js")
    assert "resName.disabled = !!s.armed ||" in js and "if (resName.disabled) resName.blur();" in js
    page = client.get("/").text
    assert '<span id="resarm" class="hint" hidden>disarm to type a name</span>' in page
    assert 'resEl("resarm").hidden = panel.hidden || !s.armed || resSaved;' in js


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
    assert "hold still" in js and "s.hand_seen" in js and "s.hand_calibrating" in js


def test_pad_draws_dead_zone_intent_dot_and_guards_the_progress_ring(client):
    js = static(client, "pad.js")
    assert "function drawPad(ctx, cfg, s)" in js
    assert "cfg.dead_zone" in js and "s.intent" in js and "hold still" in js
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


def test_arm_block_reason_sits_by_arm_and_hides_when_null_or_missing(client):
    page = client.get("/").text
    assert '<span id="armblock" hidden></span>' in page
    assert page.index('id="arm"') < page.index('id="armblock"')
    js = static(client, "app.js")
    # !undefined and !null are both true, so a snapshot without the key stays hidden
    assert '$("armblock").hidden = !s.arm_blocked;' in js
    assert '$("armblock").textContent = s.arm_blocked || "";' in js
    assert "#armblock { color: #f33;" in static(client, "style.css")


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
    }
    with_hand = TestClient(make_app(SimpleNamespace(cfg=cfg, hand=object())))
    assert with_hand.get("/config").json()["hand"] is True


def test_board_sits_in_the_hand_view_and_loads_after_race_before_app(client):
    page = client.get("/").text
    at = [page.index(i) for i in ('id="handview"', '<canvas id="tiltboard">', 'id="results"')]
    assert at == sorted(at)
    assert '<span id="handhint" hidden>hold your hand flat, palm down, then tilt</span>' in page
    at = [page.index(f"/static/{f}.js") for f in ("race", "board", "app")]
    assert at == sorted(at)
    assert "  drawPad(padCtx, cfg, s);\n  board(s, cfg);\n}" in static(client, "app.js")
    assert "#tiltboard {" in static(client, "style.css")


def test_board_draws_cmd_real_ball_bars_and_the_ghost_hand(client):
    js = static(client, "board.js")
    assert "function board(s, cfg)" in js and "cfg.tilt_full_deg" in js
    assert all(f in js for f in ("sent_x_deg", "est_x_deg", "b.valid", "s.intent.x"))
    assert '["hand", ' in js and '["cmd", ' in js and '["real", ' in js
    assert "if (g) glow(ctx, cmd, g);" in js and "performance.now()" in js
    assert '$("handhint").hidden = !on;' in js


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
    assert 'if (maze) $("cps").textContent = `0/' in app
    assert 'if (maze) $("cps").textContent = `CP ' in static(client, "hud.js")
    js = static(client, "race.js")
    assert "if (maze) {\n    if (maze.walls) walls(ctx, maze);" in js
    assert js.index("if (maze) {") < js.index("gates(ctx, maze, r, s.race)")


def test_race_js_draws_walls_under_the_race_line_only_when_the_maze_has_them(client):
    js = static(client, "race.js")
    assert "function walls(ctx, maze)" in js and "maze.walls.forEach((w) => polyline(ctx, w))" in js
    guarded = js.index("if (maze.walls) walls(ctx, maze);")
    assert guarded < js.index("ribbon(ctx, maze, cfg.ball_d_mm)")


def test_race_js_draws_the_plate_in_mm_with_a_ball_trail(client):
    js = static(client, "race.js")
    assert "function draw(ctx, cfg, maze, s)" in js
    assert "cfg.plate_inner_mm" in js and "TRAIL_S" in js and "s.ball.valid" in js


def test_race_js_draws_the_ghost_under_a_white_player(client):
    js = static(client, "race.js")
    assert "s.race" in js and "ghost_mm" in js and "#CC79A7" in js
    assert "ghost_lap_s" in js and "fmtLap" in js
    assert "#fff" in js and "#ff8c1a" not in js


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
    assert "fit(c)" in js and "err --" in js


def test_strip_err_has_a_fixed_plate_half_width_scale_and_tilt_autoscales(client):
    js = static(client, "strip.js")
    assert '"x", cfg.plate_inner_mm / 2, ERR_COLOR' in js
    assert "Math.min(p[key] / top, 1)" in js
    assert '"y", Math.max(...stripBuf.map(q => q.y)) || 1, TILT_COLOR' in js


def test_page_rings_the_tracked_ball_on_the_camera_image(client):
    page = client.get("/").text
    assert 'id="seen"' in page and 'class="view"' in page
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
