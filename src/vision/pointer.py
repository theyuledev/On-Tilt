import math

import cv2
import numpy as np

from src.control.pulse_map import center

Point = tuple[float, float]


def unwrap(angle: float, prev: float) -> float:
    # the two pointers are interchangeable so the line is only known mod 180
    while angle - prev > 90:
        angle -= 180
    while angle - prev < -90:
        angle += 180
    return angle


def horn_angle(
    gray: np.ndarray, roi: tuple[int, int, int, int] | None = None, prev: float | None = None
) -> tuple[float, Point, Point] | None:
    x0 = y0 = 0
    if roi is not None:
        x0, y0, w, h = roi
        gray = gray[y0 : y0 + h, x0 : x0 + w]
    thr = (np.percentile(gray, 50) + np.percentile(gray, 99.9)) / 2
    mask = (gray > thr).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 3:
        return None
    stats, cents = stats[1:], cents[1:]
    w, h, area = (stats[:, k] for k in (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    roundish = (area >= 20) & (area > 0.5 * w * h) & (np.minimum(w, h) > 0.5 * np.maximum(w, h))
    idx = np.flatnonzero(roundish)
    if len(idx) < 2:
        return None
    (x1, y1), (x2, y2) = cents[idx[np.argsort(-area[idx])[:2]]]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if prev is not None:
        angle = unwrap(angle, prev)
    return angle, (x1 + x0, y1 + y0), (x2 + x0, y2 + y0)


def motion_roi(a: np.ndarray, b: np.ndarray, pad: int = 40) -> tuple[int, int, int, int] | None:
    diff = cv2.absdiff(a, b)
    if diff.max() < 30:
        return None
    ys, xs = np.nonzero(diff > diff.max() / 2)
    h, w = a.shape[:2]
    x0, y0 = max(xs.min() - pad, 0), max(ys.min() - pad, 0)
    x1, y1 = min(xs.max() + pad, w), min(ys.max() + pad, h)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _first_motion(ta, a, t0, t1, thr) -> tuple[float, float] | None:
    """(delay_s, peak change) of the angle in (t0, t1], or None if it never left the noise."""
    base = a[(ta > t0 - 0.1) & (ta <= t0)]
    if len(base) == 0:
        return None
    win = (ta > t0) & (ta <= t1)
    dev = a[win] - base.mean()
    if len(dev) == 0:
        return None
    settled = dev[-max(len(dev) // 3, 1) :].mean()
    if abs(settled) < thr:
        return None
    hit = np.flatnonzero(dev * np.sign(settled) > thr)
    return ta[win][hit[0]] - t0, abs(settled)


def _jumps(tc, p, phase, want):
    i = np.flatnonzero((p[1:] != p[:-1]) & (phase[1:] == want)) + 1
    return tc[i], p[i] - p[i - 1]


def report(commands, angles, rig: dict, axis: int = 0) -> dict:
    tc = np.array([c[0] for c in commands], float)
    p = np.array([c[1 + axis] for c in commands], float)
    phase = np.array([c[3] for c in commands])
    ta = np.array([f[0] for f in angles], float)
    a = np.array([f[1] for f in angles], float)
    t_end = np.append(tc[1:], ta[-1] + 1)

    def between(name):
        lo, hi = tc[phase == name][0], t_end[phase == name][-1]
        return (ta >= lo) & (ta < hi)

    hold = a[between("hold")]
    noise = max(hold.std(), 0.02)
    thr = 3 * noise

    sw = between("sweep")
    cmd_at = np.interp(ta[sw], tc, p)
    k, b = np.polyfit(cmd_at, a[sw], 1)
    cx = center(rig)[axis]

    ts, _ = _jumps(tc, p, phase, "step")
    ends = np.append(ts[1:], ta[-1])
    moves = [_first_motion(ta, a, t0, t1, thr) for t0, t1 in zip(ts, ends, strict=True)]
    delays = [m[0] for m in moves if m]
    st = between("step")
    speed = np.abs(a[st][2:] - a[st][:-2]) / (ta[st][2:] - ta[st][:-2])

    dead = None
    rev_delays = []
    for d in sorted({int(n[3:]) for n in phase if n.startswith("rev")}):
        tr, _ = _jumps(tc, p, phase, f"rev{d}")
        both = [_first_motion(ta, a, t0, t1, thr) for t0, t1 in zip(tr[:2], tr[1:3], strict=True)]
        rev_delays += [m[0] for m in both if m]
        if dead is None and all(both):
            dead = d

    delay_ms = 1000 * float(np.median(delays)) if delays else None
    rev_lag_ms = 1000 * float(np.median(rev_delays)) if rev_delays else None
    reject = dead is None or dead > 25 or (delay_ms and rev_lag_ms and rev_lag_ms > 2 * delay_ms)
    return {
        "us_per_deg": 1 / k,
        "center_offset_deg": float(hold.mean() - (k * cx + b)),
        "noise_deg": float(noise),
        "delay_ms": delay_ms,
        "top_speed_deg_s": float(speed.max()) if len(speed) else None,
        "dead_band_us": dead,
        "reversal_lag_ms": rev_lag_ms,
        "verdict": "reject" if reject else "ok",
    }
