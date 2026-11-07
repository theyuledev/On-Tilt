import threading
import time
from pathlib import Path

import cv2

from src.io.ports import Frame


def open_camera(rig: dict) -> cv2.VideoCapture:
    exposure = rig["camera_exposure"]
    if isinstance(exposure, str):
        raise RuntimeError(f"camera_exposure not set yet: {exposure}")
    cap = cv2.VideoCapture(rig["camera_index"])
    if not cap.isOpened():
        raise RuntimeError(f"no camera at index {rig['camera_index']}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rig["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rig["camera_height"])
    cap.set(cv2.CAP_PROP_FPS, rig["camera_fps"])
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, rig["camera_manual_exposure_mode"])
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
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
