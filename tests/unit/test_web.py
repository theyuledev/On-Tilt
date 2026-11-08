import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

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
        "heartbeat_stale_s": cfg["game"]["heartbeat_stale_s"],
    }


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
