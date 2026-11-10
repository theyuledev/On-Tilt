"""An X drawn on the board with a marker, off one overhead frame."""

import math

import cv2
import numpy as np

from src.vision.maze_read import STROKES, board_mask, dark_mask
from src.vision.px_to_mm import plate_mm

RINGS = (0.6, 0.9)  # where the arms are counted, as a share of the way to the box edge
BEND_DEG = 30  # how far an arm may sit off the line of the one across from it
SQUASH_DEG = 40  # the two strokes must cross at least this wide


def arms(blob: np.ndarray, at: tuple[float, float], r: float) -> list[float]:
    """Mid angle in degrees of each run of ink met going once round a circle about `at`."""
    n = max(32, math.ceil(4 * math.pi * r))
    turn = np.arange(n) * 2 * math.pi / n
    xs = np.round(at[0] + r * np.cos(turn)).astype(int)
    ys = np.round(at[1] + r * np.sin(turn)).astype(int)
    ok = (xs >= 0) & (xs < blob.shape[1]) & (ys >= 0) & (ys < blob.shape[0])
    ink = np.zeros(n, bool)
    ink[ok] = blob[ys[ok], xs[ok]]
    if ink.all() or not ink.any():
        return []
    start = int(np.argmin(ink))  # on paper, so no run wraps round
    edges = np.flatnonzero(np.diff(np.r_[0, np.roll(ink, -start), 0]))
    return [
        (start + (a + b - 1) / 2) * 360 / n for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def four_arms(blob: np.ndarray, at: tuple[float, float]) -> bool:
    """Four arms round `at` on every ring, each in line with the one across from it."""
    reach = min(at[0], at[1], blob.shape[1] - 1 - at[0], blob.shape[0] - 1 - at[1])
    for ring in RINGS:
        a = arms(blob, at, ring * reach)
        if len(a) != 4 or any(abs(a[i + 2] - a[i] - 180) > BEND_DEG for i in (0, 1)):
            return False
        if not SQUASH_DEG <= a[1] - a[0] <= 180 - SQUASH_DEG:
            return False
    return True


def crossing(blob: np.ndarray) -> tuple[float, float] | None:
    """Where the two strokes of an X cross, in blob px, None when the blob is no X."""
    dist = cv2.distanceTransform(np.uint8(blob), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    ys, xs = np.nonzero(blob)
    # the thickest spot is the crossing on a bold X, the middle of the ink on a thin one
    for at in ((float(x), float(y)), (float(xs.mean()), float(ys.mean()))):
        if four_arms(blob, at):
            return at
    return None


def find_cross(gray: np.ndarray, px_mm: float, tune: dict = STROKES) -> tuple[float, float] | None:
    """Image px of the one letter sized X on the board, None for none or several."""
    stroke = max(1, round(tune["stroke_mm"] * px_mm))
    board = board_mask(gray, stroke, tune["board"])
    dark = dark_mask(gray, board, tune["dark"])
    lo, hi = (v * px_mm for v in tune["letter_mm"])
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.uint8(dark))
    found = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if lo <= min(w, h) and max(w, h) <= hi:
            at = crossing(lab[y : y + h, x : x + w] == i)
            if at is not None:
                found.append((x + at[0], y + at[1]))
    return found[0] if len(found) == 1 else None


def cross_mm(image: np.ndarray, vis: dict) -> tuple[float, float] | None:
    """Plate mm of the X in a full size overhead frame, mapped the way the ball is."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    at = find_cross(gray, vis["px_per_mm"])
    return plate_mm(*at, vis) if at else None
