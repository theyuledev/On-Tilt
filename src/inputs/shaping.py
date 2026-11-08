import math


def shape(x: float, y: float, cfg: dict) -> tuple[float, float]:
    """Intent -1..1 -> tilt degrees: radial dead zone, cubic expo, then gain."""
    r = math.hypot(x, y)
    dz = cfg["dead_zone"]
    if r < dz:
        return 0.0, 0.0
    # the edge of the dead zone becomes zero, the direction stays
    k = (r - dz) / (1 - dz) / r
    e, gain = cfg["expo"], cfg["intent_gain_deg"]

    def bend(v: float) -> float:
        v *= k
        return gain * ((1 - e) * v + e * v**3)

    return bend(x), bend(y)
