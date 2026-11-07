import socket
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from src.inputs.hand import HandIntent, palm_center, send_intent
from src.io.camera import find_camera, overhead_shape
from src.io.settings import ROOT, load

MODEL = ROOT / "models" / "hand_landmarker.task"


def main() -> None:
    cfg = load()
    inp = cfg["input"]
    overhead = overhead_shape(cfg["rig"])
    cap = cv2.VideoCapture(find_camera(lambda s: s != overhead, "a webcam picture"))
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )
    hand = HandIntent(inp)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"hand -> 127.0.0.1:{inp['hand_port']}, hold your palm still to set neutral")
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                continue
            t = time.monotonic()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000)
            )
            palm = palm_center(result.hand_landmarks[0]) if result.hand_landmarks else None
            send_intent(sock, hand.update(palm, t), inp["hand_port"])
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        landmarker.close()


if __name__ == "__main__":
    main()
