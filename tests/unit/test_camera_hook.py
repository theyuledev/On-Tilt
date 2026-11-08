import threading
import time

import pytest

from src.io.camera import CameraSource, ClipSource
from tests.unit.test_frames import FakeCapture, write_clip

pytestmark = pytest.mark.unit


class PacedCapture(FakeCapture):
    def read(self):
        time.sleep(0.002)
        return super().read()


def run_out(cap, process):
    src = CameraSource(cap, process=process)
    assert cap.done.wait(2)
    src._thread.join(1)
    assert cap.i == cap.n and not src.running
    src.close()
    assert cap.released
    return src


def test_camera_runs_the_hook_off_the_grab_thread_and_keeps_the_newest_result():
    calls = []

    def record(image, t):
        calls.append((int(image[0, 0]), t, threading.current_thread()))
        return ("seen", int(image[0, 0]))

    src = run_out(FakeCapture(3), record)
    values = [v for v, _, _ in calls]
    assert 1 <= len(values) <= 3 and set(values) <= {1, 2, 3}
    times = [t for _, t, _ in calls]
    assert times == sorted(times)
    assert all(th is src._hook for _, _, th in calls)
    assert src.latest() == ("seen", values[-1])
    assert src.latest() == ("seen", values[-1])
    assert src.read().image[0, 0] == 3
    assert src.read() is None


def test_a_slow_hook_skips_frames_instead_of_holding_up_the_grab():
    seen = []

    def slow(image, t):
        time.sleep(0.02)
        seen.append(int(image[0, 0]))
        return int(image[0, 0])

    src = run_out(PacedCapture(20), slow)
    assert 1 <= len(seen) < 20
    assert src.latest() == seen[-1]
    assert src.latest() == seen[-1]
    assert 0 < src.gap_s < 0.1
    assert src.hook_s >= 0.02


def test_a_broken_hook_prints_once_and_the_frames_keep_flowing(capsys):
    calls = []

    def boom(image, t):
        calls.append(t)
        raise ValueError("hook bug")

    src = run_out(PacedCapture(10), boom)
    assert calls
    assert src.latest() is None
    assert src.read().image[0, 0] == 10
    assert capsys.readouterr().err.count("ValueError: hook bug") == 1


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


def test_a_hook_that_starts_raising_keeps_the_last_good_result(tmp_path, capsys):
    path = tmp_path / "clip.avi"
    write_clip(path)
    calls = []

    def flaky(image, t):
        calls.append(t)
        if len(calls) > 1:
            raise ValueError("hook bug")
        return t

    src = ClipSource(path, process=flaky)
    first = src.read()
    while src.read() is not None:
        assert src.latest() == first.t_s
    src.close()

    assert len(calls) == 5
    assert src.latest() == first.t_s
    assert capsys.readouterr().err.count("ValueError: hook bug") == 1
