import math
from dataclasses import dataclass

import numpy as np

from src.control import ROLL

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


def _cross(t, v, level: float, i: int) -> float:
    # where v crosses level between samples i - 1 and i, linear
    return float(np.interp(level, v[i - 1 : i + 1], t[i - 1 : i + 1]))


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
    t_on = _cross(t, np.abs(seen - pre), floor, move)
    dead = float(t_on - t[k])
    final = float(np.median(seen[-max(len(seen) // 5, 1) :]))
    frac = (seen - pre) / (final - pre)
    cross = np.flatnonzero(frac[move:] >= TAU_FRAC)
    if cross.size == 0:
        return StepFit(dead, math.inf, top)
    t63 = _cross(t, frac, TAU_FRAC, int(cross[0]) + move)
    return StepFit(dead, t63 - t_on, top)


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


@dataclass(frozen=True, slots=True)
class PulseFit:
    level_us: tuple[float, float]
    us_per_deg: tuple[float, float]
    axis_deg: tuple[float, float]  # which way a + pulse on each axis rolls the ball
    r2: float


def _twice_integrated(f, dt: float):
    v = np.r_[0.0, np.cumsum(f[1:] + f[:-1]) * dt / 2]
    return np.r_[0.0, np.cumsum(v[1:] + v[:-1]) * dt / 2]


def pulse_fit(windows, dt: float, ref_us: float, g: float) -> PulseFit:
    """Fits a = A (pulse - ref_us) + b over windows of (x, y, px, py) sampled every dt, each
    window's start position and speed left free. level is where the free roll accel is zero."""
    rows, xs, ys = [], [], []
    for x, y, px, py in windows:
        tau = np.arange(len(x)) * dt
        free = np.c_[np.ones_like(tau), tau]
        drop = np.eye(len(x)) - free @ np.linalg.pinv(free)
        cols = np.c_[_twice_integrated(px - ref_us, dt), _twice_integrated(py - ref_us, dt)]
        rows.append(drop @ np.c_[cols, tau**2 / 2])
        xs.append(drop @ x)
        ys.append(drop @ y)
    X, tx, ty = np.vstack(rows), np.concatenate(xs), np.concatenate(ys)
    cx, cy = (np.linalg.lstsq(X, v, rcond=None)[0] for v in (tx, ty))
    resid = np.sum((tx - X @ cx) ** 2) + np.sum((ty - X @ cy) ** 2)
    A = np.array([cx[:2], cy[:2]])
    level = ref_us - np.linalg.solve(A, [cx[2], cy[2]])
    per_deg = ROLL * g * math.pi / 180
    return PulseFit(
        level_us=tuple(level),
        us_per_deg=tuple(per_deg / np.linalg.norm(A, axis=0)),
        axis_deg=tuple(np.degrees(np.arctan2(A[1], A[0]))),
        r2=float(1 - resid / (np.sum(tx**2) + np.sum(ty**2))),
    )


# the balance gate: a release from RELEASE_MM is inside 20 mm by RELEASE_S at each gain scale,
# then the worst radius after each AFTER_MM key stays under its value
RELEASE_MM = 60.0
RELEASE_S = {1.0: 2.5, 0.7: 3.5}
AFTER_MM = {5.0: 15.0, 15.0: 12.0}
# hold acceptance limits with no gate behind them
OVERSHOOT_PCT = 20.0
RMS_MM = 5.0


def hold_verdict(t, x, y) -> dict[str, tuple[bool, float]]:
    """t from the release on, settings gains; name -> (ok, value)."""
    t, x, y = (np.asarray(a, dtype=float) for a in (t, x, y))
    s, r = settle(t, x, y), np.hypot(x, y)
    out = {
        "t20_s": (s.t20_s <= RELEASE_S[1.0], s.t20_s),
        "overshoot_pct": (s.overshoot_pct < OVERSHOOT_PCT, s.overshoot_pct),
    }
    for after, lim in AFTER_MM.items():
        held = r[t > t[0] + after]
        worst = float(held.max()) if held.size else math.inf
        out[f"after_{after:g}_s_mm"] = (worst < lim, worst)
    out["rms_mm"] = (s.rms_mm < RMS_MM, s.rms_mm)
    return out
