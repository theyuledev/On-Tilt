from dataclasses import dataclass

from src.contracts import BallState


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
