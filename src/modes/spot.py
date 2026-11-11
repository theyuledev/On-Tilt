"""Where balance holds the ball: the centre, a point clicked on the page, or an X on the board."""

import math

from src.modes.assist import reach_mm

CENTRE = {"src": "centre", "mm": (0.0, 0.0), "found": True}


def held(cfg: dict) -> dict:
    return cfg["control"].get("target", CENTRE)


def inside(p, cfg: dict) -> tuple[float, float]:
    """p as plate mm, pulled in so the ball stays off the rim."""
    x, y = (float(v) for v in p)
    if not math.isfinite(x + y):
        raise ValueError(f"target {p!r} is not a point")
    reach = reach_mm(cfg["rig"], cfg["control"]["assist_margin_mm"])
    return min(max(x, -reach), reach), min(max(y, -reach), reach)


def aim(cfg: dict, want, cross=None) -> None:
    """want is "centre", "mouse" (stay put, the next click moves it), a clicked [x, y] in
    plate mm, or "marker" with cross = where the X was found; no X leaves the point alone."""
    now = held(cfg)
    if want == "centre":
        new = CENTRE
    elif want == "mouse":
        new = {**now, "src": "mouse", "found": True}
    elif want == "marker":
        mm = inside(cross, cfg) if cross is not None else now["mm"]
        new = {"src": "marker", "mm": mm, "found": cross is not None}
    else:
        new = {"src": "mouse", "mm": inside(want, cfg), "found": True}
    cfg["control"]["target"] = new
