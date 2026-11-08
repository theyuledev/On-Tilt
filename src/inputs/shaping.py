import math

from src.control import ROLL
from src.safety import limits


def _ease(r: float, dz: float) -> float:
    # past the dead zone: a parabola up to 2 dz then a line, smooth at both joins
    return (r - dz) ** 2 / (2 * dz) if r < 2 * dz else r - 1.5 * dz


def top_deg(cfg: dict) -> tuple[float, float]:
    """Full stick tilt per axis: the player cap, or the servo clamp if that comes first."""
    rig = cfg["rig"]
    cap = cfg["control"].get("player_max_deg", cfg["input"]["intent_gain_deg"])
    x_lo, x_hi, y_lo, y_hi = limits(rig)
    tx = min(cap, (x_hi - x_lo) / 2 / abs(rig["us_per_deg_x"]))
    ty = min(cap, (y_hi - y_lo) / 2 / abs(rig["us_per_deg_y"]))
    return tx, ty


def floor_deg(cfg: dict) -> tuple[float, float]:
    """Tilt where rolling friction just holds a resting ball."""
    c = cfg["control"]
    g = ROLL * c["sim_g_mm_s2"]
    fx, fy = (math.degrees(math.asin(c[f"rolling_friction_{a}_mm_s2"] / g)) for a in "xy")
    return fx, fy


def shape(x: float, y: float, cfg: dict) -> tuple[float, float]:
    """Intent -1..1 -> tilt degrees: smooth radial dead zone, then cubic expo from floor to top."""
    r = math.hypot(x, y)
    dz, e = cfg["input"]["dead_zone"], cfg["input"]["expo"]
    if r <= dz or r == 0:
        return 0.0, 0.0
    # full stick stays full, the direction stays
    u = _ease(r, dz) / (1 - 1.5 * dz)
    # the floor fades in over one more dead zone width so the edge has no step
    b = 1.0 if dz <= 0 else min(1.0, (r - dz) / dz)

    def bend(d: float, lo: float, hi: float) -> float:
        v = u * d
        return lo * d * b + (hi - lo) * ((1 - e) * v + e * v**3)

    (fx, fy), (tx, ty) = floor_deg(cfg), top_deg(cfg)
    return bend(x / r, fx, tx), bend(y / r, fy, ty)
