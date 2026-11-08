import math
from dataclasses import dataclass

from src.contracts import BallState
from src.control import ROLL


@dataclass(frozen=True)
class Integral:
    x: float = 0.0
    y: float = 0.0
    t_s: float | None = None


def integrate(acc: Integral, ball: BallState, c: dict) -> Integral:
    # only winds while the ball is near centre, clamped at i_max, cleared when the ball is lost
    if not ball.valid:
        return Integral()
    dt = 0.0 if acc.t_s is None else max(ball.t_s - acc.t_s, 0.0)
    lim, near = c["i_max_deg"], c["i_err_max_mm"]

    def add(v: float, pos: float) -> float:
        if abs(pos) >= near:
            return v
        return min(max(v + c["ki_deg_s_mm"] * pos * dt, -lim), lim)

    return Integral(add(acc.x, ball.x_mm), add(acc.y, ball.y_mm), ball.t_s)


def smooth(prev: float, v: float, dt: float, tau_s: float) -> float:
    # first order low pass on the velocity, a tau at or under a tick passes it straight through
    if tau_s <= dt:
        return v
    return prev + (v - prev) * dt / tau_s


def pid(pos: float, vel: float, integral: float, c: dict) -> float:
    d = min(max(c["kd_deg_s_mm"] * vel, -c["d_max_deg"]), c["d_max_deg"])
    return -(c["kp_deg_mm"] * pos + d + integral)


def _ease(u: float, lo: float, hi: float) -> float:
    # smoothstep from 0 at lo to 1 at hi, a step at hi when lo >= hi
    if hi <= lo:
        return float(u >= hi)
    t = min(max((u - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def unstick(ex: float, ey: float, vx: float, vy: float, c: dict) -> tuple[float, float]:
    # lean a bit past the rolling friction tilt toward the goal while the ball is slow and off it
    r = math.hypot(ex, ey)
    if r == 0 or c["unstick_gain"] <= 0 or c["sim_g_mm_s2"] <= 0:
        return 0.0, 0.0
    g = ROLL * c["sim_g_mm_s2"]
    f = max(c["rolling_friction_x_mm_s2"], c["rolling_friction_y_mm_s2"])
    deg = c["unstick_gain"] * math.degrees(math.asin(min(f / g, 1.0)))
    v = c["unstick_v_mm_s"]
    slow = 1 - _ease(math.hypot(vx, vy), v / 2, v)
    k = deg * _ease(r, c["unstick_in_mm"], c["unstick_full_mm"]) * slow
    return -k * ex / r, -k * ey / r
