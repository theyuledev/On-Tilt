"""A second way to find the ball, sharing nothing with the tracker: slide a dark disc on a light
ground over the picture and take the best match. It answers on an empty plate too, so it only
ever backs up or disputes a position that came from somewhere else."""

from collections.abc import Iterable

import cv2
import numpy as np


def _template(r: int) -> np.ndarray:
    # a radius of ground on every side of the disc
    side = 4 * r + 1
    out = np.full((side, side), 255, np.uint8)
    cv2.circle(out, (2 * r, 2 * r), r, 0, -1)
    return out


def disc(gray: np.ndarray, radii_px: Iterable[float]) -> tuple[float, float, float] | None:
    """Centre and radius of the best dark disc among the given sizes. None when the picture is
    too small to slide any of them over."""
    best = None
    for r in sorted({max(round(r), 1) for r in radii_px}):
        if 4 * r + 1 > min(gray.shape):
            break
        match = cv2.matchTemplate(gray, _template(r), cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(match)
        if best is None or score > best[0]:
            best = (score, x + 2.0 * r, y + 2.0 * r, float(r))
    return best and best[1:]
