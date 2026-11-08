import math
from dataclasses import dataclass

import numpy as np

TAU_FRAC = 1 - math.exp(-1)


@dataclass(frozen=True, slots=True)
class StepFit:
    dead_s: float
    tau_s: float
    top_speed_deg_s: float


@dataclass(frozen=True, slots=True)
class AccelFit:
    a_per_deg: float
    first_motion_s: float


@dataclass(frozen=True, slots=True)
class Settle:
    t20_s: float
    t5_s: float
    overshoot_pct: float
    rms_mm: float
    period_s: float


def _first_past(values, floor: float, start: int = 0) -> int | None:
    hits = np.flatnonzero(np.abs(values[start:]) > floor)
    return int(hits[0]) + start if hits.size else None


def step_response(t, sent, seen) -> StepFit:
    t, sent, seen = (np.asarray(a, dtype=float) for a in (t, sent, seen))
    steps = np.flatnonzero(sent != sent[0])
    if steps.size == 0:
        raise ValueError("sent never steps")
    k = int(steps[0])
    pre = float(np.median(seen[:k]))
    floor = max(3 * float(np.std(seen[:k])), 0.05)
    slope = (seen[2:] - seen[:-2]) / (t[2:] - t[:-2])  # 3 sample central slope
    top = float(np.max(np.abs(slope)))
    move = _first_past(seen - pre, floor, k)
    if move is None:
        return StepFit(math.inf, math.inf, top)
    final = float(np.median(seen[-max(len(seen) // 5, 1) :]))
    frac = (seen - pre) / (final - pre)
    cross = np.flatnonzero(frac[move:] >= TAU_FRAC)
    if cross.size == 0:
        return StepFit(float(t[move] - t[k]), math.inf, top)
    i = int(cross[0]) + move
    t_cross = np.interp(TAU_FRAC, frac[i - 1 : i + 1], t[i - 1 : i + 1])
    return StepFit(float(t[move] - t[k]), float(t_cross - t[move]), top)


def ball_accel(t, x, tilt_deg: float) -> AccelFit:
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    x0 = float(np.median(x[:5]))
    floor = max(3 * float(np.std(x[:5])), 0.3)
    first = _first_past(x - x0, floor)
    if first is None:
        return AccelFit(0.0, math.inf)
    win = (t >= t[first]) & (t <= t[first] + 0.5)
    # free linear term: the ball was already rolling when it cleared the floor
    a = 2 * np.polyfit(t[win] - t[first], x[win], 2)[0]
    return AccelFit(float(a) / tilt_deg, float(t[first] - t[0]))


def _stays_under(t, r, lim: float) -> float:
    out = np.flatnonzero(r >= lim)
    if out.size == 0:
        return 0.0
    k = int(out[-1]) + 1
    return float(t[k] - t[0]) if k < len(t) else math.inf


def settle(t, x, y) -> Settle:
    t, x, y = (np.asarray(a, dtype=float) for a in (t, x, y))
    r = np.hypot(x, y)
    overshoot, period = 0.0, math.inf
    if r[0] > 0:
        along = (x * x[0] + y * y[0]) / r[0]  # signed distance along the start direction
        crossings = t[np.flatnonzero(np.diff(along > 0)) + 1]
        if crossings.size >= 2:
            period = 2 * float(np.mean(np.diff(crossings)))
        overshoot = 100 * max(float(-along.min()), 0.0) / float(r[0])
    last = t >= t[-1] - 5.0
    return Settle(
        t20_s=_stays_under(t, r, 20.0),
        t5_s=_stays_under(t, r, 5.0),
        overshoot_pct=overshoot,
        rms_mm=float(np.sqrt(np.mean(r[last] ** 2))),
        period_s=period,
    )
