import math

from src.control import ROLL
from src.safety import limits


def _ease(r: float, dz: float) -> float:
    # past the dead zone: a parabola up to 2 dz then a line, smooth at both joins
    return (r - dz) ** 2 / (2 * dz) if r < 2 * dz else r - 1.5 * dz


def top_deg(cfg: dict) -> tuple[float, float]:
    """Full stick tilt per axis: the servo clamp, held to player_max_deg while guard_on."""
    rig, c = cfg["rig"], cfg["control"]
    cap = c["player_max_deg"] if c.get("guard_on", False) else math.inf
    x_lo, x_hi, y_lo, y_hi = limits(rig)
    sx, sy = abs(rig["us_per_deg_x"]), abs(rig["us_per_deg_y"])
    # a zero scale can't tilt the plate, keep it level
    tx = min(cap, (x_hi - x_lo) / 2 / sx) if sx > 0 else 0.0
    ty = min(cap, (y_hi - y_lo) / 2 / sy) if sy > 0 else 0.0
    return tx, ty


def floor_deg(cfg: dict) -> tuple[float, float]:
    """Tilt where rolling friction just holds a resting ball."""
    c = cfg["control"]
    g = ROLL * c["sim_g_mm_s2"]
    if g <= 0:
        return 0.0, 0.0
    fx, fy = (math.degrees(math.asin(min(1.0, c[f"rolling_friction_{a}_mm_s2"] / g))) for a in "xy")
    return fx, fy


def shape(x: float, y: float, cfg: dict) -> tuple[float, float]:
    """Intent -1..1 -> tilt degrees: smooth radial dead zone, then cubic expo from floor to top."""
    r = math.hypot(x, y)
    dz, e = cfg["input"]["dead_zone"], cfg["input"]["expo"]
    if r <= dz or r == 0 or dz >= 1:
        return 0.0, 0.0
    # full stick stays full, the direction stays
    u = _ease(r, dz) / _ease(1.0, dz)
    # the floor fades in over one more dead zone width so the edge has no step
    b = 1.0 if dz <= 0 else min(1.0, (r - dz) / dz)

    def bend(d: float, lo: float, hi: float) -> float:
        v = u * d
        return lo * d * b + (hi - lo) * ((1 - e) * v + e * v**3)

    (fx, fy), (tx, ty) = floor_deg(cfg), top_deg(cfg)
    return bend(x / r, fx, tx), bend(y / r, fy, ty)
