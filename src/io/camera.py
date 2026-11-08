import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2

from src.io.camera_exposure import exposure_problem
from src.io.camera_hook import Hooked
from src.io.ports import Frame

WARMUP_READS = 30
OPEN_TRIES = 3


def _first_frame_shape(cap) -> tuple[int, ...] | None:
    for _ in range(WARMUP_READS):
        ok, img = cap.read()
        if ok:
            return img.shape[:2]
    return None


def find_camera(is_wanted, what: str, video_capture=cv2.VideoCapture) -> int:
    index = 0
    while True:
        cap = video_capture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"no camera gives {what}")
        shape = _first_frame_shape(cap)
        cap.release()
        if shape is not None and is_wanted(shape):
            return index
        index += 1


def overhead_shape(rig: dict) -> tuple[int, int]:
    return rig["camera_height"], rig["camera_width"]


def set_exposure(rig: dict, run=subprocess.run) -> bool:
    tool = shutil.which("uvc-util")
    if tool is None:
        return False
    # the cam is back on auto exposure after every replug; units are 100 us
    steps = round(rig["camera_exposure_ms"] * 10)
    run([tool, "-I", "0", "-s", "auto-exposure-mode=1", "-s", f"exposure-time-abs={steps}"])
    return True


def revive_camera(run=subprocess.run) -> bool:
    tool = shutil.which("uvc-util")
    if tool is None:
        return False
    # a jammed stream comes back after a round trip through auto exposure; it has to
    # go both ways or a cam already sitting in auto sees nothing
    run([tool, "-I", "0", "-s", "auto-exposure-mode=8"])
    run([tool, "-I", "0", "-s", "auto-exposure-mode=1"])
    return True


def _open_fast(index: int, rig: dict, video_capture):
    cap = video_capture(index)
    # on mac this camera stops sending frames if width/height get set, and the
    # fast mode only kicks in if MJPG + fps are set before the first read
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, rig["camera_fps"])
    return cap


def _stream_alive(cap) -> bool:
    if _first_frame_shape(cap) is None:
        return False
    n = WARMUP_READS // 3
    return sum(cap.read()[0] for _ in range(n)) >= n / 2


def open_camera(rig: dict, video_capture=cv2.VideoCapture, run=subprocess.run):
    want = overhead_shape(rig)
    index = find_camera(lambda s: s == want, f"{want[1]}x{want[0]} frames", video_capture)
    for attempt in range(OPEN_TRIES):
        if attempt:
            revive_camera(run)
        cap = _open_fast(index, rig, video_capture)
        if _stream_alive(cap):
            set_exposure(rig, run)
            problem = exposure_problem(rig, run)
            if problem:
                print(f"camera: {problem}")
            return cap
        cap.release()
    raise RuntimeError(f"camera gives no frames after {OPEN_TRIES} opens")


class CameraSource(Hooked):
    """Grabs on its own thread and keeps only the newest frame. The hook runs on a second
    thread over the newest frame and skips the ones that land while it is busy."""

    def __init__(self, capture, process=None):
        super().__init__(process)
        self._cap = capture
        self._lock = threading.Condition()
        self._frame: Frame | None = None
        self._newest: Frame | None = None
        self.gap_s = 0.0
        self.hook_s = 0.0
        self.running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()
        self._hook = None
        if process is not None:
            self._hook = threading.Thread(target=self._hook_loop, daemon=True)
            self._hook.start()

    def _grab(self) -> None:
        while self.running:
            ok, image = self._cap.read()
            if not ok:
                self.running = False
                break
            t = time.monotonic()
            with self._lock:
                if self._newest is not None:
                    self.gap_s = t - self._newest.t_s
                self._frame = self._newest = Frame(image, t)
                self._lock.notify()
        with self._lock:
            self._lock.notify()

    def _hook_loop(self) -> None:
        done = None
        while True:
            with self._lock:
                while self._newest is done:
                    if not self.running:
                        return
                    self._lock.wait()
                frame = self._newest
            start = time.monotonic()
            result = self._run(frame.image, frame.t_s)
            with self._lock:
                self._result = result
                self.hook_s = time.monotonic() - start
            done = frame

    def read(self) -> Frame | None:
        with self._lock:
            frame, self._frame = self._frame, None
        return frame

    def close(self) -> None:
        self.running = False
        with self._lock:
            self._lock.notify()
        self._thread.join()
        if self._hook is not None:
            self._hook.join()
        self._cap.release()


class ClipSource(Hooked):
    """Replays a clip. Uses the capture times from <clip>.csv when the recorder wrote one."""

    def __init__(self, path, start: int = 0, process=None):
        super().__init__(process)
        path = Path(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(path)
        if start:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        times = path.with_suffix(".csv")
        self._times = [float(t) for t in times.read_text().split()] if times.exists() else None
        self._i = start

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        if self._times is not None:
            t = self._times[self._i]
        else:
            t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        self._i += 1
        self._result = self._run(image, t)
        return Frame(image, t)

    def close(self) -> None:
        self._cap.release()
