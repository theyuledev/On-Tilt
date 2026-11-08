import time
import tomllib
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.io.ports import Frame
from src.io.settings import ROOT, load
from src.web.server import JPEG_PERIOD_S, make_app

pytestmark = pytest.mark.unit

HUD_IDS = ("race", "timer", "speed", "cps", "banner", "ballpill", "kill", "seen", "pad", "level")
PILL_IDS = ("hand", "beat", "board", "stale")


@pytest.fixture
def client():
    return TestClient(make_app(SimpleNamespace(cfg=load())))


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
    assert '"HAND LOST"' in js and "if (!s.armed) handWasSeen = false" in js
    assert "s.armed && handWasSeen && !s.hand_seen" in js
    assert '$("level").hidden = !(s.stale && s.armed)' in js


def test_page_tracks_held_keys_by_code_so_shift_cannot_stick_one(client):
    js = static(client, "app.js")
    assert "e.code" in js and "e.key" not in js
    assert all(k in js for k in ("KeyA", "KeyD", "KeyW", "KeyS"))
    assert 'addEventListener("blur", () => held.clear())' in js


def test_c_key_sends_a_one_shot_recentre_like_arm_and_kill(client):
    assert "c re-centres the hand neutral" in client.get("/").text
    js = static(client, "app.js")
    assert 'e.code === "KeyC" && !e.metaKey && !e.ctrlKey' in js and "pending.recentre = true" in js


def test_watch_tab_sends_nothing_and_takes_no_input(client):
    js = static(client, "app.js")
    assert 'has("watch")' in js and "watch ? {} :" in js


def test_config_gives_plate_ball_and_stale_limit_from_settings(client):
    cfg = load()
    assert client.get("/config").json() == {
        "plate_inner_mm": cfg["rig"]["plate_inner_mm"],
        "ball_d_mm": cfg["rig"]["ball_d_mm"],
        "camera_width": cfg["rig"]["camera_width"],
        "heartbeat_stale_s": cfg["game"]["heartbeat_stale_s"],
        "dead_zone": cfg["input"]["dead_zone"],
    }


def test_race_maze_is_the_configured_maze_file(client):
    name = load()["game"]["maze"]
    with (ROOT / "configs" / "mazes" / f"{name}.toml").open("rb") as f:
        maze = tomllib.load(f)
    assert set(maze) == {"points", "checkpoints", "start"}
    assert client.get("/race/maze").json() == maze


def test_race_js_draws_the_plate_in_mm_with_a_ball_trail(client):
    js = static(client, "race.js")
    assert "function draw(ctx, cfg, maze, s)" in js
    assert "cfg.plate_inner_mm" in js and "TRAIL_S" in js and "s.ball.valid" in js


def test_race_js_draws_the_ghost_under_a_white_player(client):
    js = static(client, "race.js")
    assert "s.race" in js and "ghost_mm" in js and "#CC79A7" in js
    assert "ghost_lap_s" in js and "fmtLap" in js
    assert "#fff" in js and "#ff8c1a" not in js


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
