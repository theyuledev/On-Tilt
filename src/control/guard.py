import math

from src.contracts import BallState, PlateCommand
from src.control import ROLL
from src.control.pid import smooth
from src.control.pulse_map import center
from src.safety import limits


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def stop_mm(p: float, v: float, latency_s: float, brake_mm_s2: float) -> float:
    """Where the ball ends up: rolling on through the latency, then braked."""
    return p + v * latency_s + v * abs(v) / (2 * brake_mm_s2)


def need(q: float, u: float, r_out: float, latency_s: float, a_max: float) -> float:
    """Fraction of full tilt that stops an outward speed u at r_out, q being the position."""
    if u <= 0:
        return 0.0
    room = max(r_out - q - u * latency_s, 1.0)
    return u * u / (2 * room) / a_max


class Guard:
    """Caps a player's tilt and blends in a brake as the ball heads for the rim."""

    def __init__(self, cfg: dict):
        rig, c = cfg["rig"], cfg["control"]
        _, x_hi, _, y_hi = limits(rig)
        cx, cy = center(rig)
        clamp_us = (x_hi - cx, y_hi - cy)
        self.clamp = [u / rig[f"us_per_deg_{a}"] for u, a in zip(clamp_us, "xy", strict=True)]
        self.cap = [min(d, c["player_max_deg"]) for d in self.clamp]
        lag = c["servo_dead_time_s"] + cfg["vision"]["camera_latency_s"] + 1 / c["sim_tick_hz"]
        # the plate may have to swing from one end of the clamp to the other
        self.latency = [lag + 2 * u / rig["slew_us_s"] for u in clamp_us]
        g = ROLL * c["sim_g_mm_s2"]
        self.a_max = [g * math.sin(math.radians(d)) for d in self.clamp]
        self.brake = [g * math.sin(math.radians(c["guard_brake_frac"] * d)) for d in self.clamp]
        self.r_out = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2 - c["guard_margin_mm"]
        self.r_in = self.r_out - c["guard_band_mm"]
        self.c = c
        self.guard_w = [0.0, 0.0]
        self.guard_side = [0.0, 0.0]
        self.corr = [0.0, 0.0]
        self.t_s: float | None = None

    def __call__(self, ball: BallState, cmd: PlateCommand) -> PlateCommand:
        player = [
            min(max(t, -cap), cap)
            for t, cap in zip((cmd.tilt_x_deg, cmd.tilt_y_deg), self.cap, strict=True)
        ]
        if not ball.valid:
            return PlateCommand(*player)
        dt = 0.0 if self.t_s is None else max(ball.t_s - self.t_s, 0.0)
        self.t_s = ball.t_s
        pos, vel = (ball.x_mm, ball.y_mm), (ball.vx_mm_s, ball.vy_mm_s)
        return PlateCommand(*(self._axis(i, pos[i], vel[i], player[i], dt) for i in range(2)))

    def guard_view(self) -> dict:
        return {"w": [*self.guard_w], "side": [*self.guard_side]}

    def _axis(self, i: int, p: float, v: float, player: float, dt: float) -> float:
        c, T = self.c, self.latency[i]
        s = stop_mm(p, v, T, self.brake[i])
        side = 1.0 if s >= 0 else -1.0
        target = smoothstep((abs(s) - self.r_in) / (self.r_out - self.r_in))
        tau = c["guard_attack_s"] if target > self.guard_w[i] else c["guard_release_s"]
        w = self.guard_w[i] + (target - self.guard_w[i]) * min(dt / tau, 1.0)
        brake = (
            -side * self.clamp[i] * min(need(side * p, side * v, self.r_out, T, self.a_max[i]), 1.0)
        )
        if dt > 0:
            self.corr[i] = smooth(self.corr[i], w * (brake - player), dt, c["guard_lp_s"])
        self.guard_w[i], self.guard_side[i] = w, side if w > 0 else 0.0
        # never tilts further toward the rim than the player asked
        return player + self.corr[i] if side * self.corr[i] < 0 else player
