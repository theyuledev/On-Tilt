import threading

import cv2
import numpy as np
import pytest

from src.io.camera import CameraSource, ClipSource, open_camera

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


def test_camera_refuses_to_open_before_exposure_is_set():
    with pytest.raises(RuntimeError, match="not set yet"):
        open_camera({"camera_exposure": "TODO(measured: camera_exposure)"})
