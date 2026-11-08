import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app import Loop
from src.io.ports import Frame
from src.io.settings import load
from src.web.server import make_app

pytestmark = pytest.mark.unit


class FakeFrames:
    def __init__(self, *frames):
        self.frames = list(frames)
        self.running = True

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def latest(self):
        return None

    def close(self):
        pass


def test_loop_keeps_the_newest_frame():
    frame = Frame(np.zeros((8, 8, 3), np.uint8), 1.0)
    loop = Loop(load(), frames=FakeFrames(frame))
    loop.tick(1.0, 0.01)
    assert loop.frame is frame
    loop.tick(1.01, 0.01)
    assert loop.frame is frame


def test_camera_jpg_serves_the_frame_or_404():
    loop = Loop(load(), frames=FakeFrames(Frame(np.zeros((8, 8, 3), np.uint8), 1.0)))
    client = TestClient(make_app(loop))
    assert client.get("/camera.jpg").status_code == 404

    loop.tick(1.0, 0.01)
    r = client.get("/camera.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content.startswith(b"\xff\xd8")


def test_dead_grab_thread_shows_camera_lost_instead_of_the_last_frame():
    frames = FakeFrames(Frame(np.zeros((8, 8, 3), np.uint8), 1.0))
    loop = Loop(load(), frames=frames)
    client = TestClient(make_app(loop))
    loop.tick(1.0, 0.01)
    assert loop.snapshot()["camera_lost"] is False
    assert client.get("/camera.jpg").status_code == 200

    frames.running = False
    assert loop.snapshot()["camera_lost"] is True
    assert client.get("/camera.jpg").status_code == 404
    assert Loop(load()).snapshot()["camera_lost"] is False


def test_page_has_the_cam():
    client = TestClient(make_app(Loop(load())))
    assert "no camera" in client.get("/").text
    js = client.get("/static/app.js").text
    assert 'src = "/camera.jpg?' in js
    assert '"camera lost"' in js
