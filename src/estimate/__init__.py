import math
from collections import deque

import numpy as np

from src.contracts import BallState, PlateCommand
from src.sim.ball import ROLL

STEP_S = 0.01
H = np.array([1.0, 0.0])


class BallEstimator:
    """Kalman filter per axis on [pos, vel], driven by the tilts we sent run through a model of
    the servo (dead time, first order lag). predict(now) steps to when the next command actually
    moves the plate, so balance acts on where the ball will be rather than where it was."""

    def __init__(self, cfg: dict):
        self.c = cfg["control"]
        self.k = ROLL * self.c["sim_g_mm_s2"]
        self.q = self.c["est_accel_noise_mm_s2"] ** 2
        self.r = self.c["est_pos_noise_mm"] ** 2
        self.lost_after = cfg["vision"]["lost_after_misses"]
        self.cmds: deque[tuple[float, float, float]] = deque(maxlen=64)
        self.s = np.zeros((2, 2))
        self.P = np.eye(2)
        self.theta = np.zeros(2)
        self.t: float | None = None
        self.seen_s = -math.inf
        self.misses = 0
        self.skipped = False

    def command(self, t_s: float, cmd: PlateCommand) -> None:
        self.cmds.append((t_s, cmd.tilt_x_deg, cmd.tilt_y_deg))

    def update(self, t_s: float, x_mm: float, y_mm: float) -> BallState:
        self.misses = 0
        stale = t_s - self.seen_s > self.c["est_hold_s"]
        self.seen_s = t_s
        t_m = t_s - self.c["est_cam_latency_s"]
        z = np.array([x_mm, y_mm])
        if self.t is None or stale:
            self._reinit(z, t_m)
            return self.predict(t_s)
        self.s, self.P, self.theta = self._step(self.s, self.P, self.theta, self.t, t_m)
        self.t = t_m
        innov = z - self.s[:, 0]
        if np.abs(innov).max() > self.c["est_gate_mm"]:
            # one wild detection is skipped, two in a row means the ball really is there
            if self.skipped:
                self._reinit(z, t_m)
            self.skipped = not self.skipped
            return self.predict(t_s)
        self.skipped = False
        gain = self.P[:, 0] / (self.P[0, 0] + self.r)
        self.s = self.s + np.outer(innov, gain)
        v_max = self.c["est_v_max_mm_s"]
        self.s[:, 1] = np.clip(self.s[:, 1], -v_max, v_max)
        self.P = (np.eye(2) - np.outer(gain, H)) @ self.P
        return self.predict(t_s)

    def miss(self, t_s: float) -> BallState:
        self.misses += 1
        if self.misses >= self.lost_after:
            self.t = None
        return self.predict(t_s)

    def predict(self, now: float) -> BallState:
        if self.t is None or now - self.seen_s > self.c["est_hold_s"]:
            return BallState(0.0, 0.0, 0.0, 0.0, False, now)
        ahead = now + self.c["servo_dead_time_s"]
        s, _, _ = self._step(self.s, self.P, self.theta, self.t, ahead)
        (x, vx), (y, vy) = s.tolist()
        return BallState(x, y, vx, vy, True, ahead)

    def _reinit(self, z: np.ndarray, t_m: float) -> None:
        self.s = np.column_stack([z, np.zeros(2)])
        self.P = np.diag([self.r, 1e3])
        self.theta = self._sent(t_m)
        self.t = t_m
        self.skipped = False

    def _sent(self, t_s: float) -> np.ndarray:
        # the tilt the servo is heading for at t_s: the newest command past its dead time
        tilt = (0.0, 0.0)
        for t, x, y in self.cmds:
            if t > t_s - self.c["servo_dead_time_s"]:
                break
            tilt = (x, y)
        return np.array(tilt)

    def _step(self, s, P, theta, t0: float, t1: float):
        if t1 <= t0:
            return s, P, theta
        n = math.ceil((t1 - t0) / STEP_S)
        dt = (t1 - t0) / n
        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([dt * dt / 2, dt])
        Q = self.q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt * dt]])
        follow = 1 - math.exp(-dt / self.c["servo_tau_s"])
        for i in range(1, n + 1):
            theta = theta + (self._sent(t0 + i * dt) - theta) * follow
            s = s @ F.T + np.outer(self.k * np.sin(np.radians(theta)), B)
            P = F @ P @ F.T + Q
        return s, P, theta
