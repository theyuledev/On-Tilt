"""The full-rate log cut into tracks: stretches where the ball was seen in every row and never
made a step a real ball cannot make. Inside one track the tracked thing is the same thing."""

import math
from collections import Counter
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

ARRIVED = "arrived"
LOST = "ball lost"
STUCK = "stuck"
NEVER = "never reached its target"
NO_TRAVEL = "started at its target"
BROKEN = "not tracked from start to arrival"


class Track(NamedTuple):
    t: np.ndarray  # loop clock of each log row
    xy: np.ndarray  # the ball in plate mm, one row per log row


class Arrivals(NamedTuple):
    tracks: set[int]  # tracks the ball rode to a driver target on
    trials: Counter  # trials by how they ended
    stuck_unknown: int  # trials from before the driver wrote whether the ball got stuck


def _joined(a: dict, b: dict, top_mm_s: float, hole_s: float) -> bool:
    """b carries on from a: no hole the tracker could have lost the ball in and found it again,
    no step faster than a real ball."""
    if not 0 <= b["t_s"] - a["t_s"] <= hole_s:
        return False
    p, q = a["ball"], b["ball"]
    step = math.hypot(q["x_mm"] - p["x_mm"], q["y_mm"] - p["y_mm"])
    return step <= top_mm_s * (q["t_s"] - p["t_s"])


def tracks(rows: Iterable[dict], top_mm_s: float, hole_s: float) -> list[Track]:
    out: list[Track] = []
    t: list[float] = []
    xy: list[tuple[float, float]] = []
    last = None
    for row in rows:
        seen = row["ball"]["valid"]
        if t and not (seen and _joined(last, row, top_mm_s, hole_s)):
            out.append(Track(np.array(t), np.array(xy)))
            t, xy = [], []
        if seen:
            t.append(row["t_s"])
            xy.append((row["ball"]["x_mm"], row["ball"]["y_mm"]))
            last = row
    if t:
        out.append(Track(np.array(t), np.array(xy)))
    return out


def holding(tracks: list[Track], start: float, end: float) -> int | None:
    """The track that covers start to end without a break."""
    return next((i for i, tr in enumerate(tracks) if tr.t[0] <= start and end <= tr.t[-1]), None)


def _arrival(trial: dict, tracks: list[Track], at_target_mm: float) -> tuple[str, int | None]:
    if trial["ball_lost"]:
        return LOST, None
    if trial.get("stuck"):
        return STUCK, None
    if trial["reached_s"] is None:
        return NEVER, None
    start = trial["t_start"]
    i = holding(tracks, start, start + trial["reached_s"])
    if i is None:
        return BROKEN, None
    x, y = tracks[i].xy[np.searchsorted(tracks[i].t, start)]
    tx, ty = trial["target"]
    # a mark under the target is there from the start; the ball has to come from somewhere else
    if math.hypot(x - tx, y - ty) <= at_target_mm:
        return NO_TRAVEL, None
    return ARRIVED, i


def arrivals(trials: Iterable[dict], tracks: list[Track], at_target_mm: float) -> Arrivals:
    """Which tracks hold a ball that went where the driver sent it: it answers the servos and
    ends up at a target picked outside the tracker, so it is the ball."""
    held: set[int] = set()
    ends: Counter = Counter()
    unknown = 0
    for trial in trials:
        if "target" not in trial:
            continue  # the driver's closing row
        unknown += "stuck" not in trial
        end, i = _arrival(trial, tracks, at_target_mm)
        ends[end] += 1
        if i is not None:
            held.add(i)
    return Arrivals(held, ends, unknown)
