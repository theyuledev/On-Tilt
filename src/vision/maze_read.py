"""A hand-drawn maze off one camera frame: the board, the marker strokes, the S and E letters
and the corridor path between them, in image px. The tune lengths are plate mm."""

from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.maze_path import clearance, shortest_path

# how marker strokes read off a frame: stroke width, board and ink levels, letter size
STROKES = {"stroke_mm": 3.0, "board": 0.85, "dark": 0.6, "letter_mm": (7.0, 20.0)}


@dataclass(frozen=True, slots=True)
class MazeRead:
    board_box: tuple[int, int, int, int]
    s_px: tuple[float, float]
    e_px: tuple[float, float]
    walls: np.ndarray
    lines: list[np.ndarray]
    path: list[tuple[int, int]] | None
    clear_px: list[float]


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def board_mask(gray: np.ndarray, stroke_px: int, frac: float) -> np.ndarray:
    """The biggest region near the frame's brightest level, holes (strokes, letters) filled."""
    bright = np.uint8(gray > frac * np.percentile(gray, 95))
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, _disk(stroke_px))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(closed)
    if len(stats) == 1:
        raise ValueError("no bright board found in the frame")
    board = np.uint8(lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours(board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(board, cnts, -1, 1, cv2.FILLED)
    return board.astype(bool)


def dark_mask(gray: np.ndarray, board: np.ndarray, frac: float) -> np.ndarray:
    # glare is brighter than the board so it never reads as a stroke
    return (gray < frac * np.median(gray[board])) & board


def solid_mask(dark: np.ndarray, stroke_px: int) -> np.ndarray:
    """Blobs too thick to be a marker stroke (the ball, tape), rim included."""
    opened = cv2.morphologyEx(np.uint8(dark), cv2.MORPH_OPEN, _disk(stroke_px))
    return cv2.dilate(opened, _disk(stroke_px // 2 + 1)).astype(bool)


def turn_overlap(blob: np.ndarray) -> float:
    """How much a blob matches itself turned half way round: high for S, low for E."""
    big = np.uint8(cv2.resize(np.float32(blob), None, fx=4, fy=4) > 0.5)
    m = cv2.moments(big, True)
    flip = np.float32([[-1, 0, 2 * m["m10"] / m["m00"]], [0, -1, 2 * m["m01"] / m["m00"]]])
    turned = cv2.warpAffine(big, flip, big.shape[::-1])
    return float((big & turned).sum() / (big | turned).sum())


def letters(dark: np.ndarray, lo_px: float, hi_px: float):
    """S and E centres and mask: the two most compact letter-sized blobs, S the more symmetric."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(np.uint8(dark))
    found = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        if lo_px <= max(w, h) <= hi_px:
            blob = np.pad(lab[y : y + h, x : x + w] == i, max(w, h))
            stretch = cv2.HuMoments(cv2.moments(np.uint8(blob), True))[1, 0]
            found.append((stretch, i, blob))
    if len(found) < 2:
        return None, None, np.zeros_like(dark)
    pair = sorted(found, key=lambda f: f[0])[:2]
    s, e = sorted(pair, key=lambda f: -turn_overlap(f[2]))
    return tuple(cent[s[1]]), tuple(cent[e[1]]), np.isin(lab, [s[1], e[1]])


def wall_lines(walls: np.ndarray, min_px: float, tol_px: float) -> list[np.ndarray]:
    """Closed outline of each stroke at least min_px across, as (x, y) px polylines."""
    cnts, _ = cv2.findContours(np.uint8(walls), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if max(cv2.boundingRect(c)[2:]) >= min_px:
            line = cv2.approxPolyDP(c, tol_px, True).reshape(-1, 2)
            out.append(np.vstack([line, line[:1]]))
    return out


def read(gray: np.ndarray, px_mm: float, tune: dict, start=None, end=None) -> MazeRead:
    """px_mm is image px per plate mm; start / end (px) override the letter search."""
    stroke = max(1, round(tune["stroke_mm"] * px_mm))
    board = board_mask(gray, stroke, tune["board"])
    dark = dark_mask(gray, board, tune["dark"])
    dark &= ~solid_mask(dark, stroke)
    s, e, marks = letters(dark, tune["letter_mm"][0] * px_mm, tune["letter_mm"][1] * px_mm)
    s, e = start or s, end or e
    if s is None or e is None:
        raise ValueError("no S / E letters found, pass --start and --end")
    walls = dark & ~marks
    near = cv2.dilate(np.uint8(walls), _disk(round(tune["margin_mm"] * px_mm))).astype(bool)
    free = board & ~near
    clear = clearance(free)
    path = shortest_path(clear, s, e, tune["cap_mm"] * px_mm)
    ys, xs = np.nonzero(board)
    box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    lines = wall_lines(walls, tune["letter_mm"][0] * px_mm, tune["tol_mm"] * px_mm)
    return MazeRead(box, s, e, walls, lines, path, [float(clear[y, x]) for x, y in path or []])
