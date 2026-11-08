import math

from src.contracts import BallState, PlateState
from src.control import ROLL


def _axis(pos: float, vel: float, tilt_deg: float, dt: float, wall: float, g: float):
    vel += ROLL * g * math.sin(math.radians(tilt_deg)) * dt
    pos += vel * dt
    if abs(pos) > wall:
        return math.copysign(wall, pos), 0.0
    return pos, vel


def roll(ball: BallState, plate: PlateState, now: float, rig: dict, control: dict) -> BallState:
    dt = now - ball.t_s
    wall = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2
    g = control["sim_g_mm_s2"]
    x, vx = _axis(ball.x_mm, ball.vx_mm_s, plate.est_x_deg, dt, wall, g)
    y, vy = _axis(ball.y_mm, ball.vy_mm_s, plate.est_y_deg, dt, wall, g)
    return BallState(x_mm=x, y_mm=y, vx_mm_s=vx, vy_mm_s=vy, valid=True, t_s=now)
