import threading

import cv2
import numpy as np
import pytest

from src.io.camera import CameraSource, ClipSource

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


def write_clip(path, n=5):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for i in range(n):
        out.write(np.full((24, 32, 3), i * 40, np.uint8))
    out.release()


def test_clip_replays_every_frame_in_order(tmp_path):
    path = tmp_path / "clip.avi"
    write_clip(path)

    src = ClipSource(path)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.close()

    assert len(frames) == 5
    times = [f.t_s for f in frames]
    assert times == sorted(times) and times[0] < times[-1]


def test_clip_can_start_part_way_in_with_the_capture_times_lined_up(tmp_path):
    path = tmp_path / "clip.avi"
    write_clip(path)
    path.with_suffix(".csv").write_text("0.0\n0.1\n0.2\n0.3\n0.4\n")

    src = ClipSource(path, start=3)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.close()

    assert [f.t_s for f in frames] == [0.3, 0.4]
    assert frames[0].image[0, 0] == pytest.approx(120, abs=4)


def test_missing_clip_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ClipSource(tmp_path / "nope.avi")
