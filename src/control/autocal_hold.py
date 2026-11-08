import math
from bisect import bisect_left

import numpy as np

from src.control.autocal_fit import hold_accel

DIVERGED = "plate mapping looks wrong, the hold pushed the ball out"


def line(rows) -> tuple[np.ndarray, np.ndarray]:
    t, u, v = (np.array(a) for a in zip(*rows, strict=True))
    if len(t) < 3:
        return np.array([u[-1], v[-1]]), np.zeros(2)
    k, p = np.polyfit(t - t[-1], np.c_[u, v], 1)
    return p, k


def since(rows, t0: float) -> list:
    return rows[bisect_left(rows, t0, key=lambda r: r[0]) :]


class Hold:
    """PD+I centring through the probe's pulse map between doublets. step hands back the pulse
    offset, or None with "still" once the ball has settled, or None with why it gave up."""

    def __init__(self, c: dict, span_frames: int):
        self.c, self.span = c, span_frames
        self.integ = np.zeros(2)
        self.t0 = self.still = self.t_i = None

    def start(self, now: float) -> None:
        self.t0, self.still, self.t_i = now, None, None

    def step(self, now: float, rows, m) -> tuple[np.ndarray | None, str | None]:
        c = self.c
        if now - self.t0 > c["hold_timeout_s"]:
            return None, "could not hold the ball"
        p, v = line(rows[-self.span - 1 :])
        if self.t_i is not None:
            i_max = c["hold_i_max_mm_s"]
            self.integ = np.clip(self.integ + p * (rows[-1][0] - self.t_i), -i_max, i_max)
        self.t_i = rows[-1][0]
        ps, vs = line(since(rows, rows[-1][0] - c["hold_still_s"]))
        if np.hypot(*ps) < c["hold_r_mm"] and np.hypot(*vs) < c["hold_v_mm_s"]:
            self.still = now if self.still is None else self.still
            if now - self.still >= c["hold_still_s"]:
                return None, "still"
        else:
            self.still = None
        if self._diverging(now, rows, p, v):
            return None, DIVERGED
        u = np.linalg.solve(m, hold_accel(p, v, self.integ, c))
        return np.clip(u, -c["hold_max_us"], c["hold_max_us"]), None

    def _diverging(self, now: float, rows, p, v) -> bool:
        c, r = self.c, float(np.hypot(*p))
        if now - self.t0 < c["hold_diverge_s"] or r < c["hold_r_mm"] or np.dot(p, v) <= 0:
            return False
        _, u0, v0 = since(rows, now - c["hold_diverge_s"])[0]
        return r - math.hypot(u0, v0) > c["hold_diverge_mm"]
