import pytest

from src.io.camera import CameraSource, ClipSource
from tests.unit.test_frames import FakeCapture, write_clip

pytestmark = pytest.mark.unit


def test_camera_runs_the_hook_on_every_grab_and_keeps_the_newest_result():
    cap = FakeCapture(3)
    calls = []

    def record(image, t):
        calls.append((int(image[0, 0]), t))
        return ("seen", int(image[0, 0]))

    src = CameraSource(cap, process=record)
    assert cap.done.wait(1)
    src._thread.join(1)

    assert [v for v, _ in calls] == [1, 2, 3]
    times = [t for _, t in calls]
    assert times == sorted(times)
    assert src.latest() == ("seen", 3)
    assert src.latest() == ("seen", 3)
    assert src.read().image[0, 0] == 3
    assert src.read() is None
    src.close()
    assert cap.released


def test_a_broken_hook_prints_once_and_the_frames_keep_flowing(capsys):
    cap = FakeCapture(3)

    def boom(image, t):
        raise ValueError("hook bug")

    src = CameraSource(cap, process=boom)
    assert cap.done.wait(1)
    src._thread.join(1)

    assert cap.i == 3
    assert not src.running
    assert src.latest() is None
    assert src.read().image[0, 0] == 3
    assert capsys.readouterr().err.count("ValueError: hook bug") == 1
    src.close()


def test_clip_runs_the_hook_per_read_with_the_frame_time(tmp_path):
    path = tmp_path / "clip.avi"
    write_clip(path)
    calls = []

    src = ClipSource(path, process=lambda image, t: calls.append(t) or t)
    frames = []
    while (f := src.read()) is not None:
        frames.append(f)
        assert src.latest() == f.t_s
    src.close()

    assert len(frames) == 5
    assert calls == [f.t_s for f in frames]
    assert src.latest() == frames[-1].t_s
