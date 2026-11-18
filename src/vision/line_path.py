"""The centre of one marker line drawn on the board, as a path in plate mm, or why there is none."""

import math

import cv2
import numpy as np

from src.vision.line_graph import drawn, lines, links, longest_walk, walk_px
from src.vision.line_ink import ink
from src.vision.line_scribble import meets, pieces, through, zones
from src.vision.line_vote import LinePath
from src.vision.line_vote import agree as agree
from src.vision.maze_path import length
from src.vision.px_to_mm import outline_px, plate_mm
from src.vision.tracker import BallPx

NONE, MANY = "no line found", "more than one line"
BRANCH = "line splits in two: wipe one way, or put the ball on an end with one way out"


def _loose(found: list, best: list, runs: list, idx: list, zone: dict, near: float) -> bool:
    """Whether one of the runs, over chains idx, ends within near px of a walk end or at the same
    wiped scribble."""
    ends = [(idx[r[0][0]], int(not r[0][1])) for r in runs]
    ends += [(idx[r[-1][0]], int(r[-1][1])) for r in runs]
    tips = [(best[0][0], int(not best[0][1])), (best[-1][0], int(best[-1][1]))]
    return any(
        math.dist(found[a[0]].px[-a[1]], found[b[0]].px[-b[1]]) <= near
        or 0 < zone.get(a, 0) == zone.get(b)
        for a in tips
        for b in ends
    )


def best_run(found: list, wiped, cfg: dict, stroke: int, ball):
    """The longest smooth walk; the drawn length of the runs left over that are at least
    line_min_mm, with whether the longest of them hangs off the walk; and whether one of those runs
    ends within line_web_mm / 2 of a walk end or at the same wiped scribble, so the line may go on
    there. A ball given picks the walk that reaches it; then only what meets that walk is left
    over, other drawings are not."""
    reach = cfg["line_bridge_mm"] * cfg["px_per_mm"]
    bend = cfg["line_bend_deg"]
    hole = None if ball is None else (*ball[:2], ball[2] + stroke)
    web = round(cfg["line_web_mm"] * cfg["px_per_mm"] / 2)
    zone = zones(found, wiped, web)
    joined = through(zone, links(found, reach, bend, hole))
    every = [(i, e) for i in range(len(found)) for e in (0, 1)]
    cost = cfg["line_turn_mm"] * cfg["px_per_mm"]
    best = longest_walk(found, joined, bend, cost, every)
    at = []
    if ball is not None:
        at = [(i, e) for i, e in every if math.dist(found[i].px[-e], ball[:2]) <= ball[2] + stroke]
        if at and not {i for i, _ in best} & {i for i, _ in at}:
            best = longest_walk(found, joined, bend, cost, at)
    used = {i for i, _ in best}
    idx = [i for i in range(len(found)) if i not in used]
    rest = [found[i] for i in idx]
    other = lines(rest, links(rest, reach, bend))
    big = [r for r in other if drawn(rest, r) >= cfg["line_min_mm"] * cfg["px_per_mm"]]
    loose = _loose(found, best, big, idx, zone, web)
    if used & {i for i, _ in at}:
        met = meets(used, other, found, idx, zone)
        big = [r for r in big if r in met]
    if not big:
        return best, 0.0, False, loose
    walked = {n for i in used for n in found[i].nodes}
    hangs = any(n in walked for j, _ in big[0] for n in rest[j].nodes)
    return best, sum(drawn(rest, r) for r in big), hangs, loose


def resample(pts, step: float) -> list[list[float]]:
    p = np.array(pts)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    at = np.linspace(0, along[-1], max(1, round(along[-1] / step)) + 1)
    return [[round(float(np.interp(s, along, p[:, k])), 1) for k in (0, 1)] for s in at]


def find_line(
    image: np.ndarray, cfg: dict, half_mm: float, edge_mm: float, ball: BallPx | None = None
) -> LinePath:
    """cfg is the board map (the vision settings the tracker maps with); the board is read up to
    edge_mm inside the plate's half_mm square. A ball given and seen is not read as ink."""
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
    marks = np.uint8(ink(crop, cfg, 2 * stroke + 1, region) & (region > 0))
    if 2 * marks.sum() > region.sum():
        return LinePath([], NONE)
    seen = None
    if ball is not None and ball.valid:
        seen = (ball.x_px - x0, ball.y_px - y0, ball.r_px + stroke)
        cv2.circle(marks, (round(seen[0]), round(seen[1])), round(seen[2]), 0, -1)
    found, wiped = pieces(marks, stroke, cfg["line_web_mm"] * px)
    if not found:
        return LinePath([], NONE)
    best, other, hangs, loose = best_run(found, wiped, cfg, stroke, seen)
    total, chain = drawn(found, best), walk_px(found, best)
    if math.dist(chain[0], chain[-1]) < stroke and total > 4 * stroke:
        return LinePath([], BRANCH)
    if other >= cfg["line_similar"] * total:
        return LinePath([], BRANCH if hangs else MANY)
    if loose:
        return LinePath([], BRANCH)
    inside = seen is not None and cv2.pointPolygonTest(rim, seen[:2], False) >= 0
    if inside:
        near = [math.dist(p, seen[:2]) <= seen[2] + stroke for p in (chain[0], chain[-1])]
        if near[1] and not near[0]:
            chain, near = chain[::-1], near[::-1]
        chain = np.vstack(
            [[seen[:2]], chain] if near[0] else [chain, [seen[:2]]] if near[1] else [chain]
        )
    keep = cv2.approxPolyDP(np.float32(chain).reshape(-1, 1, 2), 1.0, False).reshape(-1, 2)
    mm = [plate_mm(x + x0, y + y0, cfg) for x, y in keep]
    if length(mm) < cfg["line_min_mm"]:
        return LinePath([], NONE)
    return LinePath(resample(mm, cfg["line_step_mm"]), "")
