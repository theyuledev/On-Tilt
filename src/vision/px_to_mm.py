from src.contracts import BallState
from src.vision.tracker import BallPx

SPAN = 4  # frames between the two fixes the velocity comes from; a 2-frame step is twice as noisy


def ball_state(ball: BallPx, prev: list[BallState], cfg: dict) -> BallState:
    """Plate mm from image px: origin at plate_centre_px, image y runs down so it flips."""
    cx, cy = cfg["plate_centre_px"]
    s = cfg["px_per_mm"]
    x, y = (ball.x_px - cx) / s, (cy - ball.y_px) / s
    vx = vy = 0.0
    past = [p for p in prev[-SPAN:] if p.valid]
    if ball.valid and past and ball.t_s > past[0].t_s:
        dt = ball.t_s - past[0].t_s
        vx, vy = (x - past[0].x_mm) / dt, (y - past[0].y_mm) / dt
    return BallState(x, y, vx, vy, ball.valid, ball.t_s)
