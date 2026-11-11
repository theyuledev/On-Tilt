import math

from src.contracts import BallState
from src.vision.calib import ideal_px
from src.vision.tracker import BallPx


def plate_mm(x_px: float, y_px: float, cfg: dict, lens: dict | None = None) -> tuple[float, float]:
    """Plate mm from image px: origin at plate_centre_px, image y runs down so it flips,
    then the plate's yaw (counter-clockwise, degrees) turns the axes onto the plate.
    With a lens, both points are undistorted first (still px, same K)."""
    cx, cy = cfg["plate_centre_px"]
    if lens is not None:
        (x_px, y_px), (cx, cy) = ideal_px(lens, [(x_px, y_px), (cx, cy)])
    s = cfg["px_per_mm"]
    u, v = (x_px - cx) / s, (cy - y_px) / s
    a = math.radians(cfg["plate_yaw_deg"])
    return u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)


def ball_state(
    ball: BallPx, prev: list[BallState], cfg: dict, lens: dict | None = None
) -> BallState:
    x, y = plate_mm(ball.x_px, ball.y_px, cfg, lens)
    vx = vy = 0.0
    past = [p for p in prev[-cfg["velocity_span_frames"] :] if p.valid]
    if ball.valid and past and ball.t_s > past[0].t_s:
        dt = ball.t_s - past[0].t_s
        vx, vy = (x - past[0].x_mm) / dt, (y - past[0].y_mm) / dt
    return BallState(x, y, vx, vy, ball.valid, ball.t_s)
