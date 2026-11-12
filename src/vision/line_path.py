"""The centre of one marker line drawn on the board, as a path in plate mm, or why there is none."""

import math
from collections import deque
from typing import NamedTuple

import cv2
import numpy as np

from src.vision.balls import downsample, plate_level
from src.vision.maze_path import length
from src.vision.maze_read import solid_mask
from src.vision.px_to_mm import outline_px, plate_mm
from src.vision.tracker import BallPx

NONE, MANY, OFF = "no line found", "more than one line", "line runs off the board"
BRANCH, SHORT = "line branches or crosses itself", "line too short"
STEPS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


class LinePath(NamedTuple):
    points: list[list[float]]
    reason: str


def ink(gray: np.ndarray, cfg: dict, stroke: int):
    """Marker ink: darker than line_ink x the plate around it and joined to a part darker than
    line_core, minus blobs too thick to be a stroke, which come back as the second mask."""
    level = cv2.resize(plate_level(downsample(gray, cfg), cfg, True), gray.shape[::-1])
    weak = gray < cfg["line_ink"] * level
    _, lab = cv2.connectedComponents(np.uint8(weak))
    dark = np.isin(lab, np.unique(lab[gray < cfg["line_core"] * level])) & weak
    solid = solid_mask(dark, stroke)
    return np.uint8(dark & ~solid), solid


def _blobs(mask: np.ndarray, end: bool) -> list:
    n, _, stats, cent = cv2.connectedComponentsWithStats(np.uint8(mask))
    area = stats[:, cv2.CC_STAT_AREA]
    return [(*cent[i], math.sqrt(area[i] / math.pi), end) for i in range(1, n)]


def holes(gray, solid, ball: BallPx | None, stroke: int) -> list:
    """Discs that hide the ink, (x, y, r px, a line may end there): the ball, or with no ball
    the blobs too thick to be ink, and glare that clips the camera."""
    if ball is None:
        found = _blobs(solid, True)
    else:
        found = [(ball.x_px, ball.y_px, ball.r_px + stroke, True)]
    return found + _blobs(gray == 255, False)


def bridge(marks: np.ndarray, found: list, stroke: int) -> np.ndarray:
    """Pieces of ink that run into the same hole are joined through its centre, and a piece that
    runs into the ball goes on to its centre."""
    for x, y, r, end in found:
        if end:
            cv2.circle(marks, (round(x), round(y)), round(r), 0, -1)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(marks)
    for x, y, r, end in found:
        reach = r + stroke
        x0, y0 = max(0, int(x - reach)), max(0, int(y - reach))
        win = lab[y0 : int(y + reach) + 1, x0 : int(x + reach) + 1]
        yy, xx = np.indices(win.shape)
        ring = (np.hypot(xx + x0 - x, yy + y0 - y) <= reach) & (win > 0)
        near = [k for k in np.unique(win[ring]) if stats[k, 2:4].max() >= 2 * stroke]
        if len(near) > 1 or (near and end):
            for k in near:
                m = ring & (win == k)
                at = (round(xx[m].mean()) + x0, round(yy[m].mean()) + y0)
                cv2.line(marks, at, (round(x), round(y)), 1, 3)
    return marks


def one_line(marks: np.ndarray, region: np.ndarray, stroke: int, gap: int, min_px: float):
    """The one stroke at least min_px across once small gaps are closed, or why not."""
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1,) * 2)
    closed = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, disk) & region
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed)
    big = [i for i in range(1, n) if stats[i, 2:4].max() >= min_px]
    if len(big) != 1:
        return None, MANY if big else NONE
    line = lab == big[0]
    rim = region & ~cv2.erode(region, None, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (None, OFF) if (line & rim.astype(bool)).any() else (line, "")


def _far_end(on: set, w: int, start: int):
    steps = [dy * w + dx for dx, dy in STEPS]
    prev, todo, last = {start: start}, deque([start]), start
    while todo:
        last = todo.popleft()
        for s in steps:
            if last + s in on and last + s not in prev:
                prev[last + s] = last
                todo.append(last + s)
    return last, prev


def centre_line(line: np.ndarray) -> list[tuple[int, int]]:
    """The longest walk along the stroke's skeleton, end to end, as (x, y) px."""
    on = set(np.flatnonzero(cv2.ximgproc.thinning(np.uint8(line) * 255)).tolist())
    w = line.shape[1]
    a, _ = _far_end(on, w, next(iter(on)))
    b, prev = _far_end(on, w, a)
    chain = [b]
    while chain[-1] != a:
        chain.append(prev[chain[-1]])
    return [(i % w, i // w) for i in chain]


def covers(chain, line: np.ndarray, stroke: int) -> bool:
    away = np.ones(line.shape, np.uint8)
    cv2.polylines(away, [np.int32(chain)], False, 0)
    return not (cv2.distanceTransform(away, cv2.DIST_L2, 3)[line] > stroke).any()


def resample(pts, step: float) -> list[list[float]]:
    p = np.array(pts)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    at = np.linspace(0, along[-1], max(1, round(along[-1] / step)) + 1)
    return [[round(float(np.interp(s, along, p[:, k])), 1) for k in (0, 1)] for s in at]


def find_line(
    image: np.ndarray, cfg: dict, half_mm: float, edge_mm: float, ball: BallPx | None = None
) -> LinePath:
    """cfg is the board map (the vision settings the tracker maps with); the line must stay
    edge_mm inside the plate's half_mm square. A ball given and seen is not read as ink."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    region = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(region, [np.int32(np.round(outline_px(half_mm - edge_mm, cfg)))], 1)
    x0, y0, w, h = cv2.boundingRect(region)
    stroke = round(cfg["line_stroke_mm"] * cfg["px_per_mm"])
    if min(w, h) <= 2 * stroke:
        return LinePath([], NONE)
    crop = gray[y0 : y0 + h, x0 : x0 + w]
    px = cfg["px_per_mm"]
    seen = ball if ball is not None and ball.valid else None
    if seen is not None:
        seen = seen._replace(x_px=seen.x_px - x0, y_px=seen.y_px - y0)
    region = region[y0 : y0 + h, x0 : x0 + w]
    marks, solid = ink(crop, cfg, stroke)
    marks = bridge(marks, holes(crop, solid, seen, stroke), stroke)
    gap = round(cfg["line_gap_mm"] * px / 2)
    line, why = one_line(marks, region, stroke, gap, cfg["line_min_mm"] * px / 2)
    if line is None:
        return LinePath([], why)
    chain = centre_line(line)
    if not covers(chain, line, stroke):
        return LinePath([], BRANCH)
    keep = cv2.approxPolyDP(np.float32(chain).reshape(-1, 1, 2), 1.0, False).reshape(-1, 2)
    mm = [plate_mm(x + x0, y + y0, cfg) for x, y in keep]
    if length(mm) < cfg["line_min_mm"]:
        return LinePath([], SHORT)
    return LinePath(resample(mm, cfg["line_step_mm"]), "")
