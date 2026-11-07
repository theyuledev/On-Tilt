import math
from typing import NamedTuple

import cv2
import numpy as np


class Candidate(NamedTuple):
    x_px: float
    y_px: float
    r_px: float
    score: float
    dark: bool


Box = tuple[int, int, int, int]


def _square(side: float) -> np.ndarray:
    # odd so open/close don't shift the blob by a pixel
    return np.ones((int(side) | 1,) * 2, np.uint8)


def threshold(gray: np.ndarray, cfg: dict, dark: bool) -> np.ndarray:
    """Per-pixel cut-off from the plate around it, with blobs up to the biggest ball wiped out."""
    h, w = gray.shape
    step = max(1, int(cfg["min_ball_r_px"]) // 2)
    small = cv2.resize(gray, (w // step, h // step), interpolation=cv2.INTER_LINEAR)
    square = _square(2 * cfg["max_ball_r_px"] / step + 3)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, square)
    if not dark:
        # done after the close so dark balls and lines don't break up the plate
        bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, square)
    c = cfg["min_contrast"]
    level = cv2.convertScaleAbs(bg, alpha=1 - c if dark else 1 + c)
    return cv2.resize(level, (w, h), interpolation=cv2.INTER_LINEAR)


def contrast_mask(gray: np.ndarray, cfg: dict, dark: bool, box: Box | None = None) -> np.ndarray:
    """Pixels that stand out from the plate around them, inside box (x0, y0, x1, y1)."""
    level = threshold(gray, cfg, dark)
    x0, y0, x1, y1 = box or (0, 0, gray.shape[1], gray.shape[0])
    mask = cv2.compare(gray[y0:y1, x0:x1], level[y0:y1, x0:x1], cv2.CMP_LT if dark else cv2.CMP_GT)
    # plug small glints, drop strokes too thin to hold the smallest ball, then patch notches
    # that print or a glint near the edge left in it
    r_min = cfg["min_ball_r_px"]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _square(r_min / 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _square(r_min))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _square(r_min))


def round_blobs(gray: np.ndarray, cfg: dict, dark: bool, box: Box | None = None) -> list[Candidate]:
    """Centre and radius come from the blob's area, so a stub of line stuck to it barely
    moves them. score is how much of its enclosing circle the blob fills."""
    r_min, r_max = cfg["min_ball_r_px"], cfg["max_ball_r_px"]
    mask = contrast_mask(gray, cfg, dark, box)
    offset = box[:2] if box else (0, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=offset)
    found = []
    for c in contours:
        m = cv2.moments(c)
        r = math.sqrt(m["m00"] / math.pi)
        if not r_min <= r <= r_max:
            continue
        fill = (r / cv2.minEnclosingCircle(c)[1]) ** 2
        if fill >= cfg["min_fill"]:
            found.append(Candidate(m["m10"] / m["m00"], m["m01"] / m["m00"], r, fill, dark))
    return sorted(found, key=lambda c: -c.score)


def find_balls(gray: np.ndarray, cfg: dict) -> list[Candidate]:
    """Dark balls, best first. Bright ones only when there's no dark one."""
    return round_blobs(gray, cfg, True) or round_blobs(gray, cfg, False)
