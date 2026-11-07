from typing import NamedTuple

import cv2
import numpy as np

from src.vision.balls import Candidate, find_balls, round_blobs


class BallPx(NamedTuple):
    x_px: float
    y_px: float
    r_px: float
    valid: bool
    t_s: float


class BallTracker:
    """Locks onto one ball, learns its radius and whether it's dark, coasts through short misses."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.r: float | None = None
        self.dark: bool | None = None
        self.hits = 0
        self.misses = 0
        self.track: list[tuple[float, float, float]] = []

    def predict(self, t_s: float) -> tuple[float, float]:
        t1, x1, y1 = self.track[-1]
        t0, x0, y0 = self.track[0]
        if t1 <= t0:
            return x1, y1
        k = (t_s - t1) / (t1 - t0)
        return x1 + (x1 - x0) * k, y1 + (y1 - y0) * k

    def update(self, image: np.ndarray, t_s: float) -> BallPx:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        hit = self._search(gray, t_s)
        if hit is not None:
            self.hits += 1
            self.r = hit.r_px if self.r is None else self.r + (hit.r_px - self.r) / self.hits
            self.dark = hit.dark
            self.misses = 0
            self.track = [*self.track[-1:], (t_s, hit.x_px, hit.y_px)]
            return BallPx(hit.x_px, hit.y_px, self.r, True, t_s)
        if self.r is None:
            return BallPx(0.0, 0.0, 0.0, False, t_s)
        x, y = self.predict(t_s)
        r = self.r
        self.misses += 1
        lost = self.misses >= self.cfg["lost_after_misses"]
        if lost:
            self.reset()
        return BallPx(x, y, r, not lost, t_s)

    def _search(self, gray: np.ndarray, t_s: float) -> Candidate | None:
        if self.r is None:
            found = find_balls(gray, self.cfg)
            return found[0] if found else None
        tol = self.cfg["radius_tol"]
        lo, hi = self.r * (1 - tol), self.r * (1 + tol)
        near = {**self.cfg, "max_ball_r_px": hi}
        h, w = gray.shape
        px, py = self.predict(t_s)
        px, py = min(max(px, 0), w - 1), min(max(py, 0), h - 1)
        # whole ball still fits if it lands up to a radius off the prediction
        d = 2 * hi
        box = (int(max(px - d, 0)), int(max(py - d, 0)), int(px + d), int(py + d))
        for b in (box, None):
            found = [c for c in round_blobs(gray, near, self.dark, b) if c.r_px >= lo]
            if found:
                return min(found, key=lambda c: (c.x_px - px) ** 2 + (c.y_px - py) ** 2)
        return None
