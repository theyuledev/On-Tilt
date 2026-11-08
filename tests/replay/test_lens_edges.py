import cv2
import numpy as np
import pytest

from src.io.settings import ROOT
from src.vision.calib import ideal_px, load_calib

pytestmark = pytest.mark.replay

LENS = load_calib(ROOT / "tests/fixtures/lens_rig.json")
CLIP = ROOT / "data/clips/lacrosse_ball_hand_rolled_1948.avi"
SNAP = ROOT / "data/snaps/rig_reaimed_0225.jpg"
# fixed roi per plate edge: rows y0..y1, the edge sits within 30 px of the line from x0 (at y0)
# to x1 (at y1), +1 where the plate is right of the edge (dark to bright), -1 where it's left
EDGES = {
    "lacrosse_13340_left": (CLIP, 13340, (320, 540, 258, 385, 1)),
    "reaimed_left": (SNAP, None, (260, 780, 222, 413, 1)),
    "reaimed_right": (SNAP, None, (220, 540, 935, 1046, -1)),
}


def gray(src, frame):
    if frame is None:
        return cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, image = cap.read()
    cap.release()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def edge_px(img, y0, y1, x0, x1, sign):
    # strongest step along each 4th row inside the roi
    g = cv2.GaussianBlur(img, (0, 0), 2).astype(float)
    pts = []
    for y in range(y0, y1, 4):
        left = round(x0 + (x1 - x0) * (y - y0) / (y1 - y0)) - 30
        step = sign * np.diff(g[y, left : left + 60])
        pts.append((left + int(np.argmax(step)) + 0.5, y))
    return np.array(pts)


def line_rms(p) -> float:
    c = p - p.mean(axis=0)
    normal = np.linalg.svd(c)[2][1]
    return float(np.sqrt(np.mean((c @ normal) ** 2)))


@pytest.mark.parametrize("name", EDGES)
def test_straight_plate_edges_come_out_straighter_through_the_lens(name):
    """a real lens fit: a straight plate edge bends in the raw image and straightens undistorted"""
    src, frame, roi = EDGES[name]
    if not src.exists():
        pytest.skip(f"{src.name} not on this machine")
    p = edge_px(gray(src, frame), *roi)
    raw, flat = line_rms(p), line_rms(ideal_px(LENS, p))
    print(f"{name}: raw {raw:.2f} px, undistorted {flat:.2f} px")
    assert flat < raw and flat < 1.5
