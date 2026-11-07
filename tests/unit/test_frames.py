import shutil
import threading

import cv2
import numpy as np
import pytest

from src.io.camera import CameraSource, ClipSource, open_camera, set_exposure

pytestmark = pytest.mark.unit


class FakeCapture:
    def __init__(self, n):
        self.n = n
        self.i = 0
        self.done = threading.Event()
        self.released = False

    def read(self):
        if self.i >= self.n:
            self.done.set()
            return False, None
        self.i += 1
        return True, np.full((4, 4), self.i, np.uint8)

    def release(self):
        self.released = True


def test_camera_keeps_only_the_newest_frame():
    cap = FakeCapture(50)
    src = CameraSource(cap)
    assert cap.done.wait(1)
    src._thread.join(1)
    frame = src.read()
    assert frame.image[0, 0] == 50
    assert src.read() is None
    assert not src.running
    src.close()
    assert cap.released


def test_clip_replays_every_frame_in_order(tmp_path):
    path = tmp_path / "clip.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for i in range(5):
        out.write(np.full((24, 32, 3), i * 40, np.uint8))
    out.release()

    src = ClipSource(path)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.close()

    assert len(frames) == 5
    times = [f.t_s for f in frames]
    assert times == sorted(times) and times[0] < times[-1]


def test_missing_clip_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClipSource(tmp_path / "nope.avi")


RIG = {"camera_width": 1280, "camera_height": 800, "camera_fps": 120, "camera_exposure_ms": 3.0}


def fake_cameras(shapes):
    opened = []

    class Cap:
        def __init__(self, index):
            self.shape = shapes[index] if index < len(shapes) else None
            self.sets = {}
            opened.append(self)

        def isOpened(self):
            return self.shape is not None

        def read(self):
            return True, np.zeros(self.shape, np.uint8)

        def set(self, prop, value):
            self.sets[prop] = value

        def release(self):
            pass

    return Cap, opened


def test_picks_the_camera_by_frame_size_and_sets_fast_mode():
    cap_cls, opened = fake_cameras([(1080, 1920, 3), (800, 1280, 3), (720, 1280, 3)])
    ran = []
    cap = open_camera(RIG, cap_cls, ran.append)
    assert cap.shape == (800, 1280, 3)
    assert cap.sets == {
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_FPS: 120,
    }
    assert all(not c.sets for c in opened if c is not cap)
    assert len(ran) == (1 if shutil.which("uvc-util") else 0)


def test_exposure_goes_manual_in_100us_steps(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/x/uvc-util")
    ran = []
    assert set_exposure(RIG, ran.append)
    cmd = ["/x/uvc-util", "-I", "0", "-s", "auto-exposure-mode=1", "-s", "exposure-time-abs=30"]
    assert ran == [cmd]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert not set_exposure(RIG, ran.append) and len(ran) == 1


def test_no_matching_camera_fails_clearly():
    cap_cls, _ = fake_cameras([(1080, 1920, 3), (720, 1280, 3)])
    with pytest.raises(RuntimeError, match="1280x800"):
        open_camera(RIG, cap_cls, lambda cmd: None)
