import math

from src.contracts import BallState
from src.vision.calib import load_calib, px_to_plate_mm
from src.vision.tracker import BallPx


def with_calib(cfg: dict) -> dict:
    """The vision settings, plus the lens + plate calibration when use_calib is on."""
    return {**cfg, "calib": load_calib()} if cfg["use_calib"] else cfg


def _turn(u: float, v: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a)


def plate_mm(x_px: float, y_px: float, cfg: dict) -> tuple[float, float]:
    """Plate mm from image px. Calibrated: through the lens and the plate map, then onto the
    board's centre and axes. Flat: origin at plate_centre_px, image y runs down so it flips,
    then the plate's yaw (counter-clockwise, degrees) turns the axes onto the plate."""
    calib = cfg.get("calib")
    if calib is not None:
        x, y = px_to_plate_mm([[x_px, y_px]], calib)[0]
        ox, oy = cfg["calib_centre_mm"]
        return _turn(x - ox, y - oy, cfg["calib_yaw_deg"])
    cx, cy = cfg["plate_centre_px"]
    s = cfg["px_per_mm"]
    return _turn((x_px - cx) / s, (cy - y_px) / s, cfg["plate_yaw_deg"])


def ball_state(ball: BallPx, prev: list[BallState], cfg: dict) -> BallState:
    x, y = plate_mm(ball.x_px, ball.y_px, cfg)
    vx = vy = 0.0
    past = [p for p in prev[-cfg["velocity_span_frames"] :] if p.valid]
    if ball.valid and past and ball.t_s > past[0].t_s:
        dt = ball.t_s - past[0].t_s
        vx, vy = (x - past[0].x_mm) / dt, (y - past[0].y_mm) / dt
    return BallState(x, y, vx, vy, ball.valid, ball.t_s)
