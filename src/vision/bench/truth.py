"""Where the ball is in each picture of a recorder folder, for scoring a tracker against.
The recorded position is the live tracker's own word, so it only becomes a label with something
else behind it; a picture with nothing behind it gets a reason instead."""

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from src.vision.bench.rec import Shot
from src.vision.bench.track import Track, holding

ARRIVAL = "arrived at a driver target"
SECOND = "second detector agrees"
UNSEEN = "ball not seen"
ALONE = "nothing backs the recorded ball"
DISPUTED = "detectors disagree"
UNREADABLE = "picture unreadable"

Px = tuple[float, float]
# centre and radius of what a second detector finds in the picture, None when it cannot look
Second = Callable[[Shot], tuple[float, float, float] | None]


class Label(NamedTuple):
    a: Px  # the ball at the snapshot the picture is named after
    b: Px  # and at the next one: in the picture it lies between the two
    r_px: float
    support: str
    at_rest: bool
    second: bool | None  # what the second detector said, where it ran


def _off_path(p: Px, a: Px, b: Px) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    span = dx * dx + dy * dy
    u = 0.0 if span == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span
    u = min(max(u, 0.0), 1.0)
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy)


def _slow(shot: Shot, bench: dict) -> bool:
    balls = shot.snap["ball"], shot.after["ball"]
    return all(math.hypot(b["vx_mm_s"], b["vy_mm_s"]) < bench["rest_mm_s"] for b in balls)


def _lying(found: list[Track], start: float, end: float, rest_mm: float) -> bool:
    """Tracked from start to end without leaving the spot it is on at the end."""
    i = holding(found, start, end)
    if i is None:
        return False
    t, xy = found[i]
    here = xy[min(np.searchsorted(t, end), len(t) - 1)]
    seen = xy[(t >= start) & (t <= end)]
    return float(np.hypot(*(seen - here).T).max()) < rest_mm


def _at_rest(shot: Shot, found: list[Track], bench: dict) -> bool:
    """Slow, and lying where it is for the rest time up to the picture."""
    since, now = shot.snap["t_s"] - bench["rest_s"], shot.after["t_s"]
    return _slow(shot, bench) and _lying(found, since, now, bench["rest_mm"])


def _seen(
    shot: Shot, found: list[Track], arrived: set[int], bench: dict, second: Second | None
) -> Label | str:
    k, n = shot.snap, shot.after
    if not all(s["ball"]["valid"] and s["ball_px"] for s in (k, n)):
        return UNSEEN
    (ax, ay, r), (bx, by, _) = k["ball_px"], n["ball_px"]
    agrees = None
    if second is not None:
        at = second(shot)
        if at is None:
            return UNREADABLE
        # inside the ball at the size the second detector looked for, not the tracker's
        agrees = _off_path(at[:2], (ax, ay), (bx, by)) <= at[2]
    if holding(found, k["t_s"], n["t_s"]) in arrived:
        support = ARRIVAL
    elif agrees:
        support = SECOND
    else:
        return ALONE if agrees is None else DISPUTED
    return Label((ax, ay), (bx, by), r, support, _at_rest(shot, found, bench), agrees)


def labels(
    shots: list[Shot],
    found: list[Track],
    arrived: set[int],
    bench: dict,
    second: Second | None = None,
) -> list[Label | str]:
    """One label or one reason per picture. found are the tracks of the full-rate log, arrived
    the ones that reached a driver target."""
    return [_seen(shot, found, arrived, bench, second) for shot in shots]
