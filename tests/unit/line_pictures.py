"""Board pictures drawn for the line reader tests and how close a path is to the drawn line."""

import math

import cv2
import numpy as np

from src.io.settings import load
from src.vision.line_path import find_line
from src.vision.px_to_mm import outline_px, plate_px
from src.vision.tracker import BallPx

CFG = load()
VIS, RIG = CFG["vision"], CFG["rig"]
HALF, BIG = RIG["plate_inner_mm"] / 2, RIG["ball_d_mm"] / 2
EDGE = VIS["line_edge_mm"]
TURNED = {**VIS, "plate_yaw_deg": 7.0, "plate_centre_px": [668.0, 383.0]}
STEP = VIS["line_step_mm"]

STRAIGHT = [(-90.0, -20.0), (60.0, 30.0)]
S_CURVE = [(x, 45 * math.sin(math.pi * x / 95)) for x in np.linspace(-95, 95, 60)]
CORNER = [(-80.0, -70.0), (-80.0, 60.0), (70.0, 60.0)]
# 40 px between turns, the owner's spiral
TURN_MM = 40 / VIS["px_per_mm"]
SPIRAL = [
    (r * math.cos(a), r * math.sin(a))
    for a in np.linspace(0, 4 * math.pi, 240)
    for r in [12 + TURN_MM * a / (2 * math.pi)]
]
FAINT = round(0.9 * 205)


def winding(rows=(100.0, 40.0, -20.0, -80.0), x=(-110.0, 90.0)):
    """Rows joined by half circles, first row left to right, about 1.1 m for the default."""
    pts = []
    for k, y in enumerate(rows):
        a, b = x if k % 2 == 0 else x[::-1]
        pts += [(a, y), (b, y)]
        if k + 1 < len(rows):
            r, s = (y - rows[k + 1]) / 2, 1 if k % 2 == 0 else -1
            turn = np.linspace(math.pi / 2, -math.pi / 2, 30)[1:-1]
            pts += [(b + s * r * math.cos(t), y - r + r * math.sin(t)) for t in turn]
    return pts


WINDING = winding()


def scribble(x0, top, wide, tall, step=10.0):
    """Strokes crossing step mm apart in a wide x tall box hanging below top."""
    down = [[(x0 + k, top), (x0 + k, top - tall)] for k in np.arange(0, wide + 0.1, step)]
    return down + [[(x0, top - k), (x0 + wide, top - k)] for k in np.arange(0, tall + 0.1, step)]


def split(line, x, y, gap=8.0):
    """The line in two pieces, a gap mm wide round x on its row at y."""
    i = next(k for k in range(len(line) - 1) if line[k][1] == y == line[k + 1][1])
    d = 1 if line[i + 1][0] > line[i][0] else -1
    return [line[: i + 1] + [(x - d * gap / 2, y)], [(x + d * gap / 2, y)] + line[i + 1 :]]


def lamps(img, cfg, spots):
    """Lamp reflections: white 60 px across, fading out to 130 px."""
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    out = img.astype(np.float32)
    for at in spots:
        x, y = plate_px(*at, cfg)
        out += np.clip((65 - np.hypot(xx - x, yy - y)) / 35, 0, 1) * (255 - out)
    return np.uint8(out)


def picture(cfg, lines, width=6, ball=None, cut_x_mm=None, glare=None, shade=55):
    """Grey camera frame: dark floor, board lit brighter on the right, lines drawn in plate mm."""
    img = np.full((800, 1280), 60, np.uint8)
    cv2.fillPoly(img, [np.int32(np.round(outline_px(HALF, cfg)))], 205)
    for line in lines:
        px = np.array([plate_px(x, y, cfg) for x, y in line])
        cv2.polylines(img, [np.int32(np.round(px * 16))], False, shade, width, cv2.LINE_AA, 4)
    if cut_x_mm is not None:
        a, b = plate_px(cut_x_mm, -60, cfg), plate_px(cut_x_mm, 60, cfg)
        cv2.line(img, np.int32(a), np.int32(b), 205, 4)
    if ball is not None:
        (x, y), r_mm = ball
        cv2.circle(img, np.int32(plate_px(x, y, cfg)), round(r_mm * cfg["px_per_mm"]), 25, -1)
    img = np.uint8(img * np.linspace(0.65, 1, 1280))
    if glare is not None:
        (x, y), r_px = glare
        cv2.circle(img, np.int32(plate_px(x, y, cfg)), r_px, 255, -1)
    return img


def ball_px(cfg, at, r_mm) -> BallPx:
    x, y = plate_px(*at, cfg)
    return BallPx(x, y, r_mm * cfg["px_per_mm"], True, 0.0)


def along(points, line):
    """Distance from each point to the drawn polyline and how far along it the nearest foot is."""
    p = np.array(points, float)[:, None]
    a, b = np.array(line[:-1], float), np.array(line[1:], float)
    seg = b - a
    t = np.clip(((p - a) * seg).sum(-1) / (seg * seg).sum(-1), 0, 1)
    gap = np.linalg.norm(p - a - t[..., None] * seg, axis=-1)
    k = gap.argmin(1)
    start = np.concatenate([[0], np.cumsum(np.linalg.norm(seg, axis=1))])[:-1]
    i = np.arange(len(points))
    return gap[i, k], start[k] + t[i, k] * np.linalg.norm(seg, axis=1)[k]


def follows(got, line):
    assert got.reason == ""
    assert len(got.points) >= 2
    off, arc = along(got.points, line)
    assert off.max() < 2
    a, b = got.points[0], got.points[-1]
    ends = (line[0], line[-1]), (line[-1], line[0])
    assert min(max(math.dist(a, p), math.dist(b, q)) for p, q in ends) < 5
    assert np.all(np.abs(np.abs(np.diff(arc)) - STEP) <= 1.5)


def run(img, cfg=VIS, ball=None, edge=EDGE):
    return find_line(img, cfg, HALF, edge, ball)
