import math
from collections import deque

import numpy as np

from src.contracts import BallState
from src.control import ROLL

STRIDE_S = 0.1  # a window is judged this often
GAP_S = 0.25  # a longer ball gap starts the window over
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


def responses(t: np.ndarray, xy: np.ndarray, e: np.ndarray, excite: float) -> list:
    """(push direction, ball response) for each main direction of the push e that a straight
    line can't explain by at least excite rms, from xy = p0 + v0 t + S e by least squares.
    Fitting along the main directions keeps a diagonal push, where e_x ~ e_y, well posed."""
    tt = t - t[0]
    line = np.column_stack([np.ones_like(tt), tt])
    rest = e - line @ np.linalg.lstsq(line, e, rcond=None)[0]
    _, sv, dirs = np.linalg.svd(rest, full_matrices=False)
    keep = dirs[sv / math.sqrt(len(t)) >= excite]
    if not len(keep):
        return []
    coef = np.linalg.lstsq(np.column_stack([line, e @ keep.T]), xy, rcond=None)[0]
    return list(zip(keep, coef[2:], strict=True))


def angle_deg(d: np.ndarray, r: np.ndarray) -> float:
    return math.degrees(math.atan2(d[0] * r[1] - d[1] * r[0], d @ r))


class MappingWatch:
    """Flags the axes where the ball runs off at an angle to the push of the tilts we sent,
    which is what a wrong plate yaw or a mirrored axis looks like. Only the angle counts, so
    a wrong us_per_deg doesn't trip it."""

    def __init__(self, cfg: dict):
        self.c = cfg["control"]
        rig = cfg["rig"]
        self.edge = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2 - self.c["map_wall_mm"]
        self.cmds: deque[tuple[float, float, float]] = deque()
        self.balls: deque[tuple[float, float, float]] = deque()
        self.since = self.next_s = -math.inf
        self.low = [0, 0]
        self.judged = [-math.inf, -math.inf]
        self.bad: str | None = None

    def step(self, now: float, ball: BallState, tilt: tuple[float, float]) -> bool:
        """One armed tick: the ball seen and the tilt sent. True when an axis newly looks wrong."""
        c = self.c
        self.cmds.append((now, *tilt))
        oldest = (self.balls[0][0] if self.balls else now - GAP_S) - c["servo_dead_time_s"]
        while len(self.cmds) > 1 and self.cmds[1][0] <= oldest - 3 * c["servo_tau_s"]:
            self.cmds.popleft()
        if not ball.valid or (self.balls and ball.t_s <= self.balls[-1][0]):
            return False
        if max(abs(ball.x_mm), abs(ball.y_mm)) > self.edge:
            self.balls.clear()  # a wall touch breaks the push model
            return False
        if not self.balls or ball.t_s - self.balls[-1][0] > GAP_S:
            self.balls.clear()
            self.since, self.low = ball.t_s, [0, 0]
        self.balls.append((ball.t_s, ball.x_mm, ball.y_mm))
        while self.balls[0][0] <= ball.t_s - c["map_window_s"]:
            self.balls.popleft()
        if ball.t_s - self.since < c["map_window_s"] or ball.t_s < self.next_s:
            return False
        if len(self.balls) < c["map_min_samples"]:
            return False
        self.next_s = ball.t_s + STRIDE_S
        return self._judge(ball.t_s)

    def _judge(self, now: float) -> bool:
        c = self.c
        t, x, y = (np.array(v) for v in zip(*self.balls, strict=True))
        e = pushed_mm(self.cmds, t, c)
        seen, off, flip = [False, False], [False, False], [False, False]
        for d, r in responses(t, np.column_stack([x, y]), e, c["map_excite_mm"]):
            if math.hypot(*r) < c["map_floor"]:
                continue  # the ball hardly answered, its angle is noise
            phi = abs(angle_deg(d, r))
            for k in range(2):
                if abs(d[k]) >= 0.5:
                    seen[k] = True
                    off[k] = off[k] or phi > c["map_line_deg"]
                    flip[k] = flip[k] or phi > c["map_flip_deg"]
        was = self.bad or ""
        bad = set(was)
        for k in (k for k in range(2) if seen[k]):
            if now - self.judged[k] > 2 * STRIDE_S:
                self.low[k] = 0  # two in a row means two windows close together
            self.judged[k] = now
            self.low[k] = self.low[k] + 1 if off[k] else 0
            if flip[k] or self.low[k] >= 2:
                bad.add(AXES[k])
        self.bad = "".join(a for a in AXES if a in bad) or None
        return (self.bad or "") != was
