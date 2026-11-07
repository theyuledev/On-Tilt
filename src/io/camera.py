import threading
import time
from pathlib import Path

import cv2

from src.io.ports import Frame


def _first_frame_shape(cap) -> tuple[int, ...] | None:
    for _ in range(30):
        ok, img = cap.read()
        if ok:
            return img.shape[:2]
    return None


def find_camera(rig: dict, video_capture=cv2.VideoCapture) -> int:
    want = (rig["camera_height"], rig["camera_width"])
    index = 0
    while True:
        cap = video_capture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"no camera gives {want[1]}x{want[0]} frames")
        shape = _first_frame_shape(cap)
        cap.release()
        if shape == want:
            return index
        index += 1


def open_camera(rig: dict, video_capture=cv2.VideoCapture):
    cap = video_capture(find_camera(rig, video_capture))
    # on mac this camera stops sending frames if width/height get set, and the
    # fast mode only kicks in if MJPG + fps are set before the first read
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, rig["camera_fps"])
    if _first_frame_shape(cap) is None:
        cap.release()
        raise RuntimeError("camera found but sends no frames in fast mode")
    return cap


class CameraSource:
    """Grabs on its own thread and keeps only the newest frame."""

    def __init__(self, capture):
        self._cap = capture
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self.running = True
        self._thread = threading.Thread(target=self._grab, daemon=True)
        self._thread.start()

    def _grab(self) -> None:
        while self.running:
            ok, image = self._cap.read()
            if not ok:
                self.running = False
                break
            frame = Frame(image, time.monotonic())
            with self._lock:
                self._frame = frame

    def read(self) -> Frame | None:
        with self._lock:
            frame, self._frame = self._frame, None
        return frame

    def close(self) -> None:
        self.running = False
        self._thread.join()
        self._cap.release()


class ClipSource:
    """Replays a clip. Uses the capture times from <clip>.csv when the recorder wrote one."""

    def __init__(self, path):
        path = Path(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(path)
        times = path.with_suffix(".csv")
        self._times = [float(t) for t in times.read_text().split()] if times.exists() else None
        self._i = 0

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok:
            return None
        if self._times is not None:
            t = self._times[self._i]
        else:
            t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        self._i += 1
        return Frame(image, t)

    def close(self) -> None:
        self._cap.release()
