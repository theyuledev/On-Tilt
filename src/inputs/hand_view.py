import os
import tempfile
from pathlib import Path

import cv2

from src.inputs.hand import palm_angles, palm_center

HAND_JPG = Path(tempfile.gettempdir()) / "on-tilt-hand.jpg"
JPEG_WIDTH = 640
JPEG_PERIOD_S = 1 / 15  # the model shares this process, keep the encode cheap
# same edges as mediapipe's HAND_CONNECTIONS
BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip
WHITE, GREEN, YELLOW, CYAN = (255, 255, 255), (0, 200, 0), (0, 220, 255), (255, 200, 0)


def draw(img, landmarks, neutral, angles, intent, mirror):
    out = cv2.resize(img, None, fx=JPEG_WIDTH / img.shape[1], fy=JPEG_WIDTH / img.shape[1])
    h, w = out.shape[:2]

    def px(x, y):
        return round(x * w), round(y * h)

    if landmarks is not None:
        pts = [px(p.x, p.y) for p in landmarks]
        for a, b in BONES:
            cv2.line(out, pts[a], pts[b], GREEN, 2)
        for p in pts:
            cv2.circle(out, p, 3, WHITE, -1)
        cv2.circle(out, px(*palm_center(landmarks)), 6, YELLOW, -1)
    if neutral is not None:
        cv2.drawMarker(out, px(*neutral), CYAN, cv2.MARKER_CROSS, 20, 2)
    # an overhead webcam already shows the player's view, one facing them needs a selfie flip
    # the text goes on after the flip so it reads
    if mirror:
        out = cv2.flip(out, 1)
    top = "no hand" if angles is None else f"roll {angles[0]:+.0f}  pitch {angles[1]:+.0f}"
    for i, line in enumerate((top, f"intent x {intent.x:+.2f} y {intent.y:+.2f}")):
        cv2.putText(out, line, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    return out


def frame_jpg(img, result, neutral, intent, mirror) -> bytes:
    lm = result.hand_landmarks[0] if result.hand_landmarks else None
    angles = palm_angles(result.hand_world_landmarks[0]) if lm is not None else None
    return cv2.imencode(".jpg", draw(img, lm, neutral, angles, intent, mirror))[1].tobytes()


def publish(jpg: bytes, path: Path = HAND_JPG) -> None:
    # the server reads this file, a rename means it never sees half a jpeg
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(jpg)
    os.replace(tmp, path)
