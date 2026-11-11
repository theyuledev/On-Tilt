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
        self.start(0.0)

    def start(self, now: float) -> None:
        self.t0, self.still, self.t_i, self.inward, self.bad = now, None, None, None, 0

    def step(self, now: float, rows, m) -> tuple[np.ndarray | None, str | None]:
        c = self.c
        if now - self.t0 > c["hold_timeout_s"]:
            return None, "could not hold the ball"
        p, v = line(rows[-self.span - 1 :])
        if self.t_i is not None:
            i_max = c["hold_i_max_mm_s"]
            self.integ = np.clip(self.integ + p * (rows[-1][0] - self.t_i), -i_max, i_max)
        self.t_i = rows[-1][0]
        u = np.linalg.solve(m, hold_accel(p, v, self.integ, c))
        u = np.clip(u, -c["hold_max_us"], c["hold_max_us"])
        if self._diverging(now, rows, p, v, m @ u):
            return None, DIVERGED
        ps, vs = line(since(rows, rows[-1][0] - c["hold_still_s"]))
        if np.hypot(*ps) < c["hold_r_mm"] and np.hypot(*vs) < c["hold_v_mm_s"]:
            self.still = now if self.still is None else self.still
            if now - self.still >= c["hold_still_s"]:
                return None, "still"
        else:
            self.still = None
        return u, None

    def _diverging(self, now: float, rows, p, v, a_cmd) -> bool:
        # pushing inward harder than friction + trim for longer than the chain, yet the ball
        # speeds up outward
        c, r = self.c, float(np.hypot(*p))
        n = p / r if r else p
        if r < c["hold_r_mm"] or np.dot(a_cmd, n) > -c["hold_div_accel_mm_s2"]:
            self.inward, self.bad = None, 0
            return False
        self.inward = now if self.inward is None else self.inward
        if now - self.inward < c["servo_dead_time_s"] + c["servo_tau_s"] + c["hold_div_wait_s"]:
            return False
        win = since(rows, now - c["hold_div_fit_s"])
        if len(win) < 6:  # twice the unknowns
            return False
        t, x, y = (np.array(a) for a in zip(*win, strict=True))
        a_obs = 2 * np.polyfit(t - t[-1], np.c_[x, y], 2)[0]
        out = np.dot(a_obs, n) > 0 and np.dot(v, n) > c["hold_div_v_mm_s"]
        self.bad = self.bad + 1 if out else 0
        return self.bad >= c["hold_div_frames"]
