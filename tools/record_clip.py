import sys
import time
from pathlib import Path

import cv2

from src.io.camera import CameraSource, open_camera
from src.io.ports import FrameSource
from src.io.settings import ROOT, load

CLIPS = ROOT / "data" / "clips"


def record(source: FrameSource, out: Path, seconds: float, fps: float) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    times: list[float] = []
    while not times or times[-1] - times[0] < seconds:
        frame = source.read()
        if frame is None:
            if not getattr(source, "running", True):
                break
            time.sleep(1 / fps / 4)
            continue
        if writer is None:
            h, w = frame.image.shape[:2]
            color = frame.image.ndim == 3
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h), color)
        writer.write(frame.image)
        times.append(frame.t_s)
    if writer is not None:
        writer.release()
    out.with_suffix(".csv").write_text("".join(f"{t:.6f}\n" for t in times))
    return len(times)


def main() -> None:
    name, seconds = sys.argv[1], float(sys.argv[2])
    rig = load()["rig"]
    source = CameraSource(open_camera(rig))
    try:
        n = record(source, CLIPS / f"{name}.avi", seconds, rig["camera_fps"])
    finally:
        source.close()
    print(f"{n} frames -> {CLIPS / name}.avi")


if __name__ == "__main__":
    main()
