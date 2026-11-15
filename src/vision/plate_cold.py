"""Finds the board anywhere in the picture, with no start place and no markers: the plate-sized
square, at any place and turn, whose edges are brighter just inside than just outside. Works in a
coarse top view in the calibration's mm, one pixel per plate_edge_width_mm, where the lens is
gone; the edge tracker then has to lock on from the corners found, so a sheet of paper or a lamp
spot is never taken for the board."""

import math

import cv2
import numpy as np

from src.vision.calib import plate_mm_to_px, px_to_plate_mm
from src.vision.plate_find import SQUARE, PlateTracker


def top_maps(cfg: dict, half_mm: float, shape):
    """Where each top-view pixel sits in the frame, which of them the frame shows, and the view's
    corner in mm. The view reaches a board's width either way of the frame's middle."""
    h, w = shape[:2]
    step, span = cfg["plate_edge_width_mm"], 2 * half_mm
    cx, cy = px_to_plate_mm([(w / 2, h / 2)], cfg["calib"])[0]
    gx, gy = np.meshgrid(
        np.arange(cx - span, cx + span, step), np.arange(cy - span, cy + span, step)
    )
    mm = np.column_stack([gx.ravel(), gy.ravel()])
    px = plate_mm_to_px(mm, cfg["calib"])
    # far off the lens model folds back into the frame: only pixels that map there and back count
    ok = np.abs(px_to_plate_mm(px, cfg["calib"]) - mm).max(axis=1) < step
    ok &= (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    x, y = (np.ascontiguousarray(v, np.float32).reshape(gx.shape) for v in px.T)
    return x, y, ok.reshape(gx.shape).astype(np.float32), np.array([cx - span, cy - span])


def turn_step(cfg: dict, half_mm: float) -> float:
    """The coarse search's turn step, radians: it moves a corner half the edge tracker's reach."""
    return math.atan(cfg["plate_search_mm"] / 2 / half_mm)


def _box(total, pad, n, x0, x1, y0, y1):
    """Sum over [u + x0, u + x1) x [v + y0, v + y1) for every centre (u, v) of the n x n view."""
    a, b, c, d = pad + y0, pad + y1, pad + x0, pad + x1
    return (
        total[b : b + n, d : d + n]
        - total[a : a + n, d : d + n]
        - total[b : b + n, c : c + n]
        + total[a : a + n, c : c + n]
    )


def _strips(s: int, b: int, m: int):
    """Per edge of a square of half side s, the strip b deep just inside it and the one just
    outside, as (x0, x1, y0, y1) from the middle, m short of the corners like the edge tracker."""
    along = (m - s, s - m)
    return (
        ((s - b, s, *along), (s, s + b, *along)),
        ((-s, b - s, *along), (-s - b, -s, *along)),
        ((*along, s - b, s), (*along, s, s + b)),
        ((*along, -s, b - s), (*along, -s - b, -s)),
    )


def _turned(top, ok, deg, s, b, m, level, cfg):
    """Over the view turned by deg: per centre, how strongly the square's edges step down from
    inside to outside, 0 unless three are clear. The fourth may lie against something as bright."""
    n = top.shape[0]
    rot = cv2.getRotationMatrix2D((n / 2, n / 2), deg, 1.0)
    pad = s + b  # as far as a strip reaches from the middle
    turned = [cv2.warpAffine(v, rot, (n, n)) for v in (top, ok)]
    tot, cnt = (
        cv2.integral(cv2.copyMakeBorder(t, *[pad] * 4, cv2.BORDER_CONSTANT)) for t in turned
    )
    need = cfg["plate_edge_min_frac"] * b * 2 * (s - m)
    score, clear = np.zeros((n, n)), np.zeros((n, n), np.int32)
    for inner, outer in _strips(s, b, m):
        ni, no = _box(cnt, pad, n, *inner), _box(cnt, pad, n, *outer)
        mi = _box(tot, pad, n, *inner) / np.maximum(ni, 1)
        mo = _box(tot, pad, n, *outer) / np.maximum(no, 1)
        full = (ni >= need) & (no >= need)
        score += np.where(full, np.clip((mi - mo) / level, 0, 1), 0)
        clear += full & (mi - mo >= cfg["plate_edge_step"] * mi)
    return rot, np.where(clear >= 3, score, 0)


def squares(top: np.ndarray, ok: np.ndarray, cfg: dict, half_mm: float) -> list[np.ndarray]:
    """Per turn the plate-sized square with the strongest edges; best first. View px."""
    step, reach = cfg["plate_edge_width_mm"], cfg["plate_search_mm"]
    s, b, m = round(half_mm / step), round(reach / 2 / step), round(reach / step)
    level = float(np.median(top[ok > 0]))
    if level <= 0:
        return []
    out = []
    # a square repeats every quarter turn
    for deg in np.arange(-45, 45, math.degrees(turn_step(cfg, half_mm))):
        rot, score = _turned(top, ok, deg, s, b, m, level, cfg)
        v, u = np.unravel_index(score.argmax(), score.shape)
        if score[v, u] > 0:
            sq = np.c_[s * SQUARE + (u, v), np.ones(4)] @ cv2.invertAffineTransform(rot).T
            out.append((float(score[v, u]), sq))
    return [q for _, q in sorted(out, key=lambda c: -c[0])]


def _side(lit: np.ndarray, pts: np.ndarray, out: np.ndarray, deep: int, sign: int) -> np.ndarray:
    """Mean of the view 1 to deep px to one side of every point: sign -1 inside, +1 outside."""
    n = lit.shape[0]
    at = [np.clip(np.rint(pts + sign * d * out).astype(int), 0, n - 1) for d in range(1, deep + 1)]
    return np.mean([lit[xy[..., 1], xy[..., 0]] for xy in at], axis=0)


def _edge(lit, p, q, cfg, r, deep, slopes):
    """The line near edge p-q with the strongest bright-to-dark step across it, as a point and a
    direction; its outward normal; whether the step is clear. Lines are tried at every whole
    pixel out up to r either way, at each of the slopes."""
    samples = cfg["plate_edge_samples"]
    mid, t = (p + q) / 2, (q - p) / np.linalg.norm(q - p)
    # the corners run counter-clockwise, so this side is out
    out = np.array([t[1], -t[0]])
    end = np.linalg.norm(q - p) / 2 - r
    a = np.linspace(-end, end, samples)
    offs = np.arange(-r, r + 1.0)
    pts = (mid + a[:, None] * t) + (offs[:, None, None] + slopes[:, None] * a)[..., None] * out
    inside = _side(lit, pts, out, deep, -1)
    diff = inside - _side(lit, pts, out, deep, 1)
    enough = np.isfinite(diff).sum(axis=-1) >= cfg["plate_edge_min_frac"] * samples
    drop = np.full(enough.shape, -np.inf)
    drop[enough] = np.nanmedian(diff[enough], axis=-1)
    o, k = np.unravel_index(drop.argmax(), drop.shape)
    clear = bool(enough[o, k]) and drop[o, k] >= cfg["plate_edge_step"] * np.nanmedian(inside[o, k])
    return (mid + offs[o] * out, t + slopes[k] * out), out, clear


def _cross(a, b) -> np.ndarray:
    (p, d), (q, e) = a, b
    return p + np.linalg.solve(np.column_stack([d, -e]), q - p)[0] * d


def refine(
    top: np.ndarray, ok: np.ndarray, quad: np.ndarray, cfg: dict, half_mm: float
) -> np.ndarray | None:
    """Each edge moved and turned on its own to the strongest step near it, within the edge
    tracker's reach; an edge that shows none sits a board's width from the one across. None
    under three edges. View px."""
    step, reach = cfg["plate_edge_width_mm"], cfg["plate_search_mm"]
    lit = np.where(ok > 0, top, np.nan)
    r, deep = round(reach / step), round(reach / 2 / step)
    # half the coarse turn step, out to two of them: a tipped board's edges are not quite square
    slopes = np.tan(np.arange(-2, 2.5, 0.5) * turn_step(cfg, half_mm))
    fits = [_edge(lit, quad[i], quad[(i + 1) % 4], cfg, r, deep, slopes) for i in range(4)]
    if sum(clear for _, _, clear in fits) < 3:
        return None
    lines = [line for line, _, _ in fits]
    for i, (_, _, clear) in enumerate(fits):
        if not clear:
            (p, d), out, _ = fits[(i + 2) % 4]
            lines[i] = (p - 2 * half_mm / step * out, d)
    return np.array([_cross(lines[i - 1], lines[i]) for i in range(4)])


class ColdFinder:
    """find() gives an edge tracker locked on the board, or None when no board shows."""

    def __init__(self, cfg: dict, half_mm: float):
        self.cfg, self.half_mm, self.maps = cfg, half_mm, None

    def find(self, image: np.ndarray, near: np.ndarray) -> PlateTracker | None:
        """near: the corners last known. The board looks the same every quarter turn, so the
        corners found are named after the nearest of them."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.maps is None:
            self.maps = top_maps(self.cfg, self.half_mm, gray.shape)
        x, y, ok, origin = self.maps
        top = cv2.remap(gray, x, y, cv2.INTER_LINEAR).astype(np.float32) * ok
        for quad in squares(top, ok, self.cfg, self.half_mm):
            fine = refine(top, ok, quad, self.cfg, self.half_mm)
            if fine is None:
                continue
            mm = fine * self.cfg["plate_edge_width_mm"] + origin
            seed = min(
                (np.roll(mm, k, axis=0) for k in range(4)),
                key=lambda c: np.linalg.norm(c - near, axis=1).sum(),
            )
            found = PlateTracker(self.cfg, self.half_mm, seed)
            # twice: the first look re-centres the search when the seed was far off
            for _ in range(2):
                found.update(image)
            if not found.lost and found.seen >= 3:
                return found
        return None
