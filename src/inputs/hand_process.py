import socket
import time
from functools import partial

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from src.inputs.hand import HandIntent, palm_center, recentre_asked, send_intent
from src.io.camera import WARMUP_READS, CameraSource, find_camera, overhead_shape
from src.io.settings import ROOT, load

MODEL = ROOT / "models" / "hand_landmarker.task"


def run(
    frames, detect, hand: HandIntent, send, recentre=lambda: False, clock=time.monotonic
) -> None:
    """Runs the model on the newest frame; frames that queued up while it ran are dropped."""
    while frames.running:
        frame = frames.read()
        if frame is None:
            time.sleep(0.001)
            continue
        now = clock()
        intent = hand.update(detect(frame.image, now), now, recentre())
        send(
            intent,
            seen=hand.seen,
            calibrating=hand.calibrating,
            progress=hand.progress,
            lag_s=now - frame.t_s,
        )


def main() -> None:
    cfg = load()
    inp = cfg["input"]
    overhead = overhead_shape(cfg["rig"])
    cap = cv2.VideoCapture(find_camera(lambda s: s != overhead, "a webcam picture"))
    # a fresh capture can miss its first reads, and the grab thread stops on a miss
    if not any(cap.read()[0] for _ in range(WARMUP_READS)):
        raise RuntimeError("webcam sends no frames")
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )

    def detect(img, t: float) -> tuple[float, float] | None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000)
        )
        return palm_center(result.hand_landmarks[0]) if result.hand_landmarks else None

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.setblocking(False)
    frames = CameraSource(cap)
    print(f"hand -> 127.0.0.1:{inp['hand_port']}, hold still to set neutral, c re centres")
    try:
        run(
            frames,
            detect,
            HandIntent(inp),
            partial(send_intent, sock, port=inp["hand_port"]),
            partial(recentre_asked, sock),
        )
    except KeyboardInterrupt:
        pass
    finally:
        frames.close()
        landmarker.close()


if __name__ == "__main__":
    main()
