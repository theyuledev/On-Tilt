"""The route a ball follows over a marker drawing on the board, as a path in plate mm, or why there
is none."""

import math

import cv2
import numpy as np

from src.vision.line_graph import drawn, lines, links, longest_walk, walk_px
from src.vision.line_ink import ink
from src.vision.line_scribble import pieces, through, zones
from src.vision.line_vote import LinePath
from src.vision.line_vote import agree as agree
from src.vision.maze_path import length
from src.vision.px_to_mm import outline_px, plate_mm
from src.vision.tracker import BallPx

NONE = "no line found"


def best_run(found: list, wiped, cfg: dict, stroke: int, ball):
    """The longest smooth walk, and the runs left out of it that are at least line_min_mm, as px
    polylines. A ball given picks the walk that reaches it."""
    reach = cfg["line_bridge_mm"] * cfg["px_per_mm"]
    bend = cfg["line_bend_deg"]
    hole = None if ball is None else (*ball[:2], ball[2] + stroke)
    zone = zones(found, wiped, round(cfg["line_web_mm"] * cfg["px_per_mm"] / 2))
    joined = through(zone, links(found, reach, bend, hole))
    every = [(i, e) for i in range(len(found)) for e in (0, 1)]
    cost = cfg["line_turn_mm"] * cfg["px_per_mm"]
    best = longest_walk(found, joined, bend, cost, every)
    if ball is not None:
        at = [(i, e) for i, e in every if math.dist(found[i].px[-e], ball[:2]) <= ball[2] + stroke]
        if at and not {i for i, _ in best} & {i for i, _ in at}:
            best = longest_walk(found, joined, bend, cost, at)
    used = {i for i, _ in best}
    rest = [c for i, c in enumerate(found) if i not in used]
    big = cfg["line_min_mm"] * cfg["px_per_mm"]
    return best, [
        walk_px(rest, r) for r in lines(rest, links(rest, reach, bend)) if drawn(rest, r) >= big
    ]


def _strokes(marks: np.ndarray, cfg: dict, stroke: int) -> list:
    """The runs of marks at least line_min_mm long, as px polylines."""
    found, _ = pieces(marks, stroke)
    big = cfg["line_min_mm"] * cfg["px_per_mm"]
    runs = lines(
        found, links(found, cfg["line_bridge_mm"] * cfg["px_per_mm"], cfg["line_bend_deg"])
    )
    return [walk_px(found, r) for r in runs if drawn(found, r) >= big]


def _turned(walk: list) -> list:
    return [(i, not fwd) for i, fwd in walk[::-1]]


def seam(found: list, walk: list, stroke: int) -> int | None:
    """Where in walk_px the walk's end comes back to a point it already passed: the first time it
    was at the same fork (forks joined by a chain under stroke px are one), else the first point
    within stroke px that is more than 4 strokes back. None when it does not."""
    root: dict = {}

    def same(n):
        while n in root:
            n = root[n]
        return n

    for c in found:
        if c.length < stroke and same(c.nodes[0]) != same(c.nodes[1]):
            root[same(c.nodes[1])] = same(c.nodes[0])
    stops = np.cumsum([0] + [len(found[i].px) for i, _ in walk])
    at = [
        (stops[m] + e * (stops[m + 1] - stops[m] - 1), n)
        for m, (i, fwd) in enumerate(walk)
        for e, n in enumerate(found[i].nodes[:: 1 if fwd else -1])
    ]
    again = [j for j, n in at[:-1] if same(n) == same(at[-1][1])]
    if again:
        return int(again[0])
    chain = walk_px(found, walk)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(chain, axis=0), axis=1))])
    near = (np.linalg.norm(chain - chain[-1], axis=1) <= stroke) & (along < along[-1] - 4 * stroke)
    return int(np.argmax(near)) if near.any() else None


def resample(pts, step: float) -> list[list[float]]:
    p = np.array(pts)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    at = np.linspace(0, along[-1], max(1, round(along[-1] / step)) + 1)
    return [[round(float(np.interp(s, along, p[:, k])), 1) for k in (0, 1)] for s in at]


def _ring_start(ring: list) -> list:
    """A closed ring started at its point lowest in x + y, so every picture starts it alike."""
    s = min(range(len(ring) - 1), key=lambda j: ring[j][0] + ring[j][1])
    return ring[s:-1] + ring[: s + 1]


def find_line(
    image: np.ndarray, cfg: dict, half_mm: float, edge_mm: float, ball: BallPx | None = None
) -> LinePath:
    """cfg is the board map (the vision settings the tracker maps with); the board is read up to
    edge_mm inside the plate's half_mm square. A ball given and seen is not read as ink; on a free
    end of the walk it is the path's start. A walk that comes back to a point it passed ends
    exactly on that point (last point == an earlier one), unless that lap is under
    line_loop_min_mm; a closed loop with no ball on it starts at its lowest x + y. Runs left out
    of the walk and faint ink apart from it come back as skipped."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    region = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(region, [np.int32(np.round(outline_px(half_mm - edge_mm, cfg)))], 1)
    x0, y0, w, h = cv2.boundingRect(region)
    px = cfg["px_per_mm"]
    stroke = round(cfg["line_stroke_mm"] * px)
    if min(w, h) <= 2 * stroke:
        return LinePath([], NONE)
    crop = gray[y0 : y0 + h, x0 : x0 + w]
    region = region[y0 : y0 + h, x0 : x0 + w]
    rim = np.float32(np.array(outline_px(half_mm - edge_mm, cfg)) - (x0, y0))
    marks, faint = ink(crop, cfg, 2 * stroke + 1, region)
    marks = np.uint8(marks & (region > 0))
    if 2 * marks.sum() > region.sum():
        return LinePath([], NONE)
    # faint ink apart from the stroke, not the stroke's own soft edge
    faint = np.uint8(faint & (region > 0)) & (
        1 - cv2.dilate(marks, np.ones((2 * stroke + 1,) * 2, np.uint8))
    )
    seen = None
    if ball is not None and ball.valid:
        seen = (ball.x_px - x0, ball.y_px - y0, ball.r_px + stroke)
        cv2.circle(marks, (round(seen[0]), round(seen[1])), round(seen[2]), 0, -1)
    found, wiped = pieces(marks, stroke, cfg["line_web_mm"] * px)
    if not found:
        return LinePath([], NONE)
    walk, left = best_run(found, wiped, cfg, stroke, seen)
    inside = seen is not None and cv2.pointPolygonTest(rim, seen[:2], False) >= 0
    tips = walk_px(found, walk)[[0, -1]]
    near = [inside and math.dist(p, seen[:2]) <= seen[2] + stroke for p in tips]
    back = _turned(walk)
    if (near[1] and not near[0]) or (
        not near[0] and seam(found, walk, stroke) is None and seam(found, back, stroke) is not None
    ):
        walk, near = back, near[::-1]
    chain, k = walk_px(found, walk), seam(found, walk, stroke)
    if near[0]:
        chain = np.vstack([[seen[:2]], chain])
        k = 0 if near[1] else None if k is None else k + 1

    def mm(c):
        keep = cv2.approxPolyDP(np.float32(c).reshape(-1, 1, 2), 1.0, False).reshape(-1, 2)
        return [plate_mm(x + x0, y + y0, cfg) for x, y in keep]

    # simplified open: a polyline whose two ends coincide would simplify to nothing
    lap = None if k is None else mm(chain[k:])
    lap = lap and lap + lap[:1]
    skipped = [resample(mm(r), cfg["line_step_mm"]) for r in left + _strokes(faint, cfg, stroke)]
    if lap is None or length(lap) < cfg["line_loop_min_mm"]:
        whole = mm(chain)
        if length(whole) < cfg["line_min_mm"]:
            return LinePath([], NONE)
        return LinePath(resample(whole, cfg["line_step_mm"]), "", skipped)
    ring = resample(lap, cfg["line_step_mm"])
    if not k:
        return LinePath(ring if near[0] else _ring_start(ring), "", skipped)
    return LinePath(resample(mm(chain[: k + 1]), cfg["line_step_mm"])[:-1] + ring, "", skipped)
