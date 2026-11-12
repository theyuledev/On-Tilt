"""The centre of one marker line drawn on the board, as a path in plate mm, or why there is none."""

import math
from typing import NamedTuple

import cv2
import numpy as np

from src.vision.balls import downsample, plate_level
from src.vision.line_graph import chains, drawn, lines, links, longest_walk, walk_px
from src.vision.maze_path import length
from src.vision.px_to_mm import outline_px, plate_mm
from src.vision.tracker import BallPx

NONE, MANY = "no line found", "more than one line"
BRANCH, SHORT = "line branches or crosses itself", "line too short"


class LinePath(NamedTuple):
    points: list[list[float]]
    reason: str


def ink(gray: np.ndarray, cfg: dict, wide: int) -> np.ndarray:
    """Thin marks darker than the board right around them, kept where part of them is clearly
    dark. wide (px) is more than the broadest stroke."""
    b = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide, wide))
    board = cv2.resize(plate_level(downsample(gray, cfg), cfg, False), gray.shape[::-1])
    # capped just above the bare board so a reflection's edge can't pass for ink
    shut = np.minimum(cv2.morphologyEx(b, cv2.MORPH_CLOSE, disk), (1 + cfg["line_glare"]) * board)
    dark = 1 - b / np.maximum(shut, 1.0)
    weak = dark > cfg["line_weak"]
    _, lab = cv2.connectedComponents(np.uint8(weak))
    marks = np.uint8(np.isin(lab, np.unique(lab[dark > cfg["line_core"]])) & weak)
    gap = round(cfg["line_gap_mm"] * cfg["px_per_mm"] / 2)
    joint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1,) * 2)
    return cv2.morphologyEx(marks, cv2.MORPH_CLOSE, joint) > 0


def pieces(marks: np.ndarray, stroke: int, web: float = 0.0):
    """Chains of the marks' skeleton, with specks and short spurs dropped. Where three forks or
    more sit within web px of each other (a scribble, not a line) the marks are wiped and read
    again; the wiped area comes back too (None when nothing was wiped)."""
    found = chains(cv2.ximgproc.thinning(np.uint8(marks) * 255) > 0, 3 * stroke)
    degree: dict[int, int] = {}
    at = {}
    for c in found:
        for e, node in enumerate(c.nodes):
            degree[node] = degree.get(node, 0) + 1
            at[node] = c.px[-e]
    forks = np.array([at[n] for n, d in degree.items() if d >= 3]).reshape(-1, 2)
    close = np.linalg.norm(forks[:, None] - forks[None], axis=-1) <= web
    wipe = forks[close.sum(1) >= 3] if web else forks[:0]
    if len(wipe):
        wiped = np.zeros_like(marks)
        for x, y in wipe:
            cv2.circle(wiped, (round(x), round(y)), round(web / 2), 1, -1)
        return pieces(marks & (1 - wiped), stroke)[0], wiped
    return [c for c in found if c.length >= stroke or min(degree[n] for n in c.nodes) > 1], None


def through(found: list, wiped: np.ndarray, reach: int, joined: dict) -> dict:
    """In each wiped scribble, the two longest chains that end within reach px of it are joined
    through it."""
    _, zones = cv2.connectedComponents(cv2.dilate(wiped, np.ones((2 * reach + 1,) * 2, np.uint8)))
    h, w = zones.shape
    by: dict[int, list] = {}
    for i, c in enumerate(found):
        for e in (0, 1):
            x, y = np.clip(np.round(c.px[-e]).astype(int), 0, [w - 1, h - 1])
            if zones[y, x]:
                by.setdefault(zones[y, x], []).append((c.length, i, e))
    out = dict(joined)
    for ends in by.values():
        top = sorted(ends, reverse=True)
        if len(top) >= 2:
            a, b = top[0][1:], top[1][1:]
            out[a], out[b] = b, a
    return out


def best_run(found: list, wiped, cfg: dict, stroke: int, ball):
    """The longest smooth walk, and the drawn length of the longest run left over with whether
    it hangs off the walk. A ball given picks the walk that reaches it; that walk is the line
    whatever else is drawn."""
    reach = cfg["line_bridge_mm"] * cfg["px_per_mm"]
    bend = cfg["line_bend_deg"]
    hole = None if ball is None else (*ball[:2], ball[2] + stroke)
    joined = links(found, reach, bend, hole)
    if wiped is not None:
        joined = through(found, wiped, round(cfg["line_web_mm"] * cfg["px_per_mm"] / 2), joined)
    every = [(i, e) for i in range(len(found)) for e in (0, 1)]
    cost = cfg["line_turn_mm"] * cfg["px_per_mm"]
    best = longest_walk(found, joined, bend, cost, every)
    if ball is not None:
        at = [(i, e) for i, e in every if math.dist(found[i].px[-e], ball[:2]) <= ball[2] + stroke]
        if at and not {i for i, _ in best} & {i for i, _ in at}:
            best = longest_walk(found, joined, bend, cost, at)
        if {i for i, _ in best} & {i for i, _ in at}:
            return best, 0.0, False
    used = {i for i, _ in best}
    idx = [i for i in range(len(found)) if i not in used]
    rest = [found[i] for i in idx]
    other = lines(rest, links(rest, reach, bend))
    if not other:
        return best, 0.0, False
    walked = {n for i in used for n in found[i].nodes}
    hangs = any(n in walked for j, _ in other[0] for n in rest[j].nodes)
    return best, drawn(rest, other[0]), hangs


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
    marks = np.uint8(ink(crop, cfg, 2 * stroke + 1) & (region > 0))
    if 2 * marks.sum() > region.sum():
        return LinePath([], NONE)
    seen = None
    if ball is not None and ball.valid:
        seen = (ball.x_px - x0, ball.y_px - y0, ball.r_px + stroke)
        cv2.circle(marks, (round(seen[0]), round(seen[1])), round(seen[2]), 0, -1)
    found, wiped = pieces(marks, stroke, cfg["line_web_mm"] * px)
    # more pieces than fit the board a bridge apart is noise, not a drawing
    if (
        not found
        or len(found) * cfg["line_bridge_mm"] * cfg["line_stroke_mm"] > region.sum() / px**2
    ):
        return LinePath([], NONE)
    best, other, hangs = best_run(found, wiped, cfg, stroke, seen)
    total, chain = drawn(found, best), walk_px(found, best)
    if math.dist(chain[0], chain[-1]) < stroke and total > 4 * stroke:
        return LinePath([], BRANCH)
    if other >= cfg["line_similar"] * total:
        return LinePath([], BRANCH if hangs else MANY)
    inside = seen is not None and cv2.pointPolygonTest(rim, seen[:2], False) >= 0
    if inside:
        near = [math.dist(p, seen[:2]) <= seen[2] + stroke for p in (chain[0], chain[-1])]
        chain = np.vstack(
            [[seen[:2]], chain] if near[0] else [chain, [seen[:2]]] if near[1] else [chain]
        )
    keep = cv2.approxPolyDP(np.float32(chain).reshape(-1, 1, 2), 1.0, False).reshape(-1, 2)
    mm = [plate_mm(x + x0, y + y0, cfg) for x, y in keep]
    if length(mm) < cfg["line_min_mm"]:
        return LinePath([], SHORT)
    return LinePath(resample(mm, cfg["line_step_mm"]), "")
