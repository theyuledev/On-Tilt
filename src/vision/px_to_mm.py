import math

from src.contracts import BallState
from src.vision.tracker import BallPx


def ball_state(ball: BallPx, prev: list[BallState], cfg: dict) -> BallState:
    """Plate mm from image px: origin at plate_centre_px, image y runs down so it flips,
    then the plate's yaw (counter-clockwise, degrees) turns the axes onto the plate."""
    cx, cy = cfg["plate_centre_px"]
    s = cfg["px_per_mm"]
    u, v = (ball.x_px - cx) / s, (cy - ball.y_px) / s
    a = math.radians(cfg["plate_yaw_deg"])
    x, y = u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)
    vx = vy = 0.0
    past = [p for p in prev[-cfg["velocity_span_frames"] :] if p.valid]
    if ball.valid and past and ball.t_s > past[0].t_s:
        dt = ball.t_s - past[0].t_s
        vx, vy = (x - past[0].x_mm) / dt, (y - past[0].y_mm) / dt
    return BallState(x, y, vx, vy, ball.valid, ball.t_s)
