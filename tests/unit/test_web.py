import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.contracts import BallState, Intent, PlateCommand, PlateState, RunLogRecord, ServoCommand
from src.games import ghost
from src.io.ports import Frame
from src.io.settings import load
from src.web.server import JPEG_PERIOD_S, make_app

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    return TestClient(make_app(SimpleNamespace(cfg=load())))


def test_page_has_ball_heartbeat_and_board_lost(client):
    page = client.get("/").text
    assert 'id="ball"' in page
    assert 'id="beat"' in page
    assert 'id="board"' in page
    assert 'id="stale"' in page and "no input" in page


def test_page_has_a_hand_pill_keyed_on_the_snapshot_flags(client):
    page = client.get("/").text
    assert 'id="hand"' in page and "hold still" in page
    assert "s.hand_seen" in page and "s.hand_calibrating" in page


def test_page_tracks_held_keys_by_code_so_shift_cannot_stick_one(client):
    page = client.get("/").text
    assert "e.code" in page and "e.key" not in page
    assert all(k in page for k in ("KeyA", "KeyD", "KeyW", "KeyS"))
    assert 'addEventListener("blur", () => held.clear())' in page


def test_config_gives_plate_ball_and_stale_limit_from_settings(client):
    cfg = load()
    assert client.get("/config").json() == {
        "plate_inner_mm": cfg["rig"]["plate_inner_mm"],
        "ball_d_mm": cfg["rig"]["ball_d_mm"],
        "camera_width": cfg["rig"]["camera_width"],
        "heartbeat_stale_s": cfg["game"]["heartbeat_stale_s"],
    }


def test_page_rings_the_tracked_ball_on_the_camera_image(client):
    page = client.get("/").text
    assert 'id="seen"' in page and 'class="view"' in page
    assert "s.ball_px" in page and "cfg.camera_width" in page


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


def ghost_record(t, x, y):
    return RunLogRecord(
        t_s=t,
        mode="hand",
        armed=True,
        ball=BallState(x_mm=x, y_mm=y, vx_mm_s=0.0, vy_mm_s=0.0, valid=True, t_s=t),
        intent=Intent(x=0.0, y=0.0, valid=True, t_s=t),
        plate_cmd=PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0),
        servo_cmd=ServoCommand(x_us=1500, y_us=1500),
        plate_state=PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0),
    )


def test_race_ghost_is_empty_without_a_ghost(client):
    assert client.get("/race/ghost").json() == []


def test_race_ghost_serves_the_trace_of_the_saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr("src.web.server.ROOT", tmp_path)
    recs = [ghost_record(5.0, 0.0, 0.0), ghost_record(5.5, 10.0, -4.0)]
    ghost.save(recs, tmp_path / "data" / "ghost.jsonl")
    cfg = load()
    cfg["game"]["ghost"] = "data/ghost.jsonl"
    client = TestClient(make_app(SimpleNamespace(cfg=cfg)))
    assert client.get("/race/ghost").json() == [[0.0, 0.0, 0.0], [0.5, 10.0, -4.0]]


def test_race_ghost_404s_when_the_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.web.server.ROOT", tmp_path)
    cfg = load()
    cfg["game"]["ghost"] = "data/nope.jsonl"
    assert TestClient(make_app(SimpleNamespace(cfg=cfg))).get("/race/ghost").status_code == 404
