import math

from src.contracts import BallState, PlateState
from src.control import ROLL


def _rub(vx: float, vy: float, fx: float, fy: float, dt: float) -> tuple[float, float]:
    # rolling friction along the velocity, f per axis, never past zero, so a ball sticks under it
    speed = math.hypot(vx, vy)
    if speed == 0:
        return 0.0, 0.0
    return vx * max(1 - fx * dt / speed, 0.0), vy * max(1 - fy * dt / speed, 0.0)


def _wall(pos: float, vel: float, wall: float) -> tuple[float, float]:
    if abs(pos) > wall:
        return math.copysign(wall, pos), 0.0
    return pos, vel


def roll(ball: BallState, plate: PlateState, now: float, rig: dict, control: dict) -> BallState:
    dt = now - ball.t_s
    wall = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2
    k = ROLL * control["sim_g_mm_s2"] * dt
    vx = ball.vx_mm_s + k * math.sin(math.radians(plate.est_x_deg))
    vy = ball.vy_mm_s + k * math.sin(math.radians(plate.est_y_deg))
    fx, fy = control["rolling_friction_x_mm_s2"], control["rolling_friction_y_mm_s2"]
    vx, vy = _rub(vx, vy, fx, fy, dt)
    x, vx = _wall(ball.x_mm + vx * dt, vx, wall)
    y, vy = _wall(ball.y_mm + vy * dt, vy, wall)
    return BallState(x_mm=x, y_mm=y, vx_mm_s=vx, vy_mm_s=vy, valid=True, t_s=now)
