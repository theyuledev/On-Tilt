import math


def _ease(r: float, dz: float) -> float:
    # past the dead zone: a parabola up to 2 dz then a line, smooth at both joins
    return (r - dz) ** 2 / (2 * dz) if r < 2 * dz else r - 1.5 * dz


def shape(x: float, y: float, cfg: dict) -> tuple[float, float]:
    """Intent -1..1 -> tilt degrees: smooth radial dead zone, cubic expo, then gain."""
    r = math.hypot(x, y)
    dz = cfg["dead_zone"]
    if r < dz:
        return 0.0, 0.0
    # full stick stays full, the direction stays
    k = _ease(r, dz) / (1 - 1.5 * dz) / r
    e, gain = cfg["expo"], cfg["intent_gain_deg"]

    def bend(v: float) -> float:
        v *= k
        return gain * ((1 - e) * v + e * v**3)

    return bend(x), bend(y)
