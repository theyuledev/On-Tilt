import math
from collections import deque

import numpy as np

from src.contracts import BallState
from src.control import ROLL

STRIDE_S = 0.1  # a window is judged this often, a longer ball gap starts the window over
STEP_S = 0.001
AXES = "xy"


def pushed_mm(cmds, t: np.ndarray, c: dict) -> np.ndarray:
    """Per axis double integral of ROLL g sin(tilt) at the times t, the tilt being the sent
    (t, x_deg, y_deg) commands through the servo's dead time and first order lag."""
    dead, tau = c["servo_dead_time_s"], c["servo_tau_s"]
    grid = np.arange(t[0] - 3 * tau, t[-1] + STEP_S, STEP_S)
    sent_t = np.array([cmd[0] for cmd in cmds])
    sent = np.array([cmd[1:] for cmd in cmds])
    i = np.searchsorted(sent_t, grid - dead, side="right") - 1
    u = np.where(i[:, None] >= 0, sent[np.maximum(i, 0)], 0.0)
    f = 1 - math.exp(-STEP_S / tau)
    decay = (1 - f) ** np.arange(1, len(grid) + 1)
    theta = np.column_stack(
        [u[0, k] * decay + f * np.convolve(u[:, k], decay / (1 - f))[: len(grid)] for k in range(2)]
    )
    e = np.cumsum(np.cumsum(ROLL * c["sim_g_mm_s2"] * np.sin(np.radians(theta)), 0), 0)
    return np.column_stack([np.interp(t, grid, e[:, k] * STEP_S**2) for k in range(2)])


def fit(t: np.ndarray, pos: np.ndarray, e: np.ndarray) -> tuple[float, float]:
    """s of pos = p0 + v0 t + s e by least squares, and the rms of e a straight line can't
    explain (how much the window says about s at all)."""
    tt = t - t[0]
    line = np.column_stack([np.ones_like(tt), tt])
    rest = e - line @ np.linalg.lstsq(line, e, rcond=None)[0]
    s = np.linalg.lstsq(np.column_stack([line, e]), pos, rcond=None)[0][2]
    return float(s), float(np.sqrt(np.mean(rest**2)))


class MappingWatch:
    """Flags the axes where the ball doesn't move the way the sent tilts push it, which is what
    a wrong plate yaw or a mirrored axis looks like."""

    def __init__(self, cfg: dict):
        self.c = cfg["control"]
        self.cmds: deque[tuple[float, float, float]] = deque()
        self.balls: deque[tuple[float, float, float]] = deque()
        self.since = self.next_s = -math.inf
        self.low = [0, 0]
        self.bad: str | None = None

    def step(self, now: float, ball: BallState, tilt: tuple[float, float]) -> bool:
        """One armed tick: the ball seen and the tilt sent. True when an axis newly looks wrong."""
        self.cmds.append((now, *tilt))
        if not ball.valid or (self.balls and ball.t_s <= self.balls[-1][0]):
            return False
        if not self.balls or ball.t_s - self.balls[-1][0] > STRIDE_S:
            self.balls.clear()
            self.since, self.low = ball.t_s, [0, 0]
        self.balls.append((ball.t_s, ball.x_mm, ball.y_mm))
        win = self.c["map_window_s"]
        while self.balls[0][0] <= ball.t_s - win:
            self.balls.popleft()
        oldest = self.balls[0][0] - self.c["servo_dead_time_s"] - 3 * self.c["servo_tau_s"]
        while len(self.cmds) > 1 and self.cmds[1][0] <= oldest:
            self.cmds.popleft()
        if ball.t_s - self.since < win or ball.t_s < self.next_s or len(self.balls) <= 3:
            return False
        self.next_s = ball.t_s + STRIDE_S
        return self._judge()

    def _judge(self) -> bool:
        t, x, y = (np.array(v) for v in zip(*self.balls, strict=True))
        e = pushed_mm(self.cmds, t, self.c)
        was = self.bad or ""
        now = set(was)
        for k, pos in enumerate((x, y)):
            s, excite = fit(t, pos, e[:, k])
            if excite < self.c["map_excite_mm"]:
                continue
            self.low[k] = self.low[k] + 1 if s < self.c["map_s_bad"] else 0
            if s < self.c["map_s_flip"] or self.low[k] >= 2:
                now.add(AXES[k])
        self.bad = "".join(a for a in AXES if a in now) or None
        return (self.bad or "") != was
