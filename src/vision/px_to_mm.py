import math

from src.contracts import BallState
from src.vision.calib import load_calib, plate_mm_to_px, px_to_plate_mm
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


def plate_px(x_mm: float, y_mm: float, cfg: dict) -> tuple[float, float]:
    """Image px from plate mm, the other way round."""
    calib = cfg.get("calib")
    if calib is not None:
        u, v = _turn(x_mm, y_mm, -cfg["calib_yaw_deg"])
        ox, oy = cfg["calib_centre_mm"]
        x, y = plate_mm_to_px([[u + ox, v + oy]], calib)[0]
        return float(x), float(y)
    u, v = _turn(x_mm, y_mm, -cfg["plate_yaw_deg"])
    cx, cy = cfg["plate_centre_px"]
    s = cfg["px_per_mm"]
    return cx + s * u, cy - s * v


def outline_px(half_mm: float, cfg: dict, per_side: int = 12) -> list[list[float]]:
    """A square of that half size around the plate centre, as image px, walked corner to corner.
    Several points a side, the lens bends a straight edge."""
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    out = []
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1], strict=True):
        for i in range(per_side):
            t = i / per_side
            x, y = plate_px(half_mm * (ax + (bx - ax) * t), half_mm * (ay + (by - ay) * t), cfg)
            out.append([round(x, 1), round(y, 1)])
    return out


def ball_state(ball: BallPx, prev: list[BallState], cfg: dict) -> BallState:
    x, y = plate_mm(ball.x_px, ball.y_px, cfg)
    vx = vy = 0.0
    past = [p for p in prev[-cfg["velocity_span_frames"] :] if p.valid]
    if ball.valid and past and ball.t_s > past[0].t_s:
        dt = ball.t_s - past[0].t_s
        vx, vy = (x - past[0].x_mm) / dt, (y - past[0].y_mm) / dt
    return BallState(x, y, vx, vy, ball.valid, ball.t_s)
