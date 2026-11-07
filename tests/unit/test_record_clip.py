import numpy as np
import pytest

from src.io.camera import ClipSource
from src.io.ports import Frame
from tools.record_clip import record

pytestmark = pytest.mark.unit


class FakeCamera:
    def __init__(self, n, dt):
        self.frames = [
            Frame(np.full((24, 32, 3), i * 10 % 255, np.uint8), 100.0 + i * dt) for i in range(n)
        ]
        self.running = True

    def read(self):
        if not self.frames:
            self.running = False
            return None
        return self.frames.pop(0)


def test_records_for_the_asked_time_and_replays_with_capture_times(tmp_path):
    out = tmp_path / "clips" / "centre.avi"
    n = record(FakeCamera(100, 0.01), out, seconds=0.5, fps=100)
    assert n == 51

    src = ClipSource(out)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
    src.close()

    assert len(frames) == n
    assert frames[0].t_s == pytest.approx(100.0)
    assert frames[-1].t_s == pytest.approx(100.5)


def test_stops_when_the_camera_dies(tmp_path):
    n = record(FakeCamera(5, 0.01), tmp_path / "short.avi", seconds=10, fps=100)
    assert n == 5
