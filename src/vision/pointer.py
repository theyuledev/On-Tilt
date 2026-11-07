import itertools
import math

import cv2
import numpy as np

from src.control.pulse_map import center
from src.vision.plate_pose import phase_mask, responses

Point = tuple[float, float]


def unwrap(angle: float, prev: float) -> float:
    # the two pointers are interchangeable so the line is only known mod 180
    while angle - prev > 90:
        angle -= 180
    while angle - prev < -90:
        angle += 180
    return angle


def _blobs(gray: np.ndarray, min_area: int = 200):
    """(labels, label ids, boxes, centroids) of the bright square-ish blobs."""
    thr = (np.percentile(gray, 50) + np.percentile(gray, 99.9)) / 2
    n, labels, stats, cents = cv2.connectedComponentsWithStats((gray > thr).astype(np.uint8))
    if n < 2:
        return labels, np.empty(0, int), np.empty((0, 4), int), np.empty((0, 2))
    stats, cents = stats[1:], cents[1:]
    w, h, area = (stats[:, k] for k in (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    ok = (area >= min_area) & (area > 0.5 * w * h) & (np.minimum(w, h) > 0.5 * np.maximum(w, h))
    return labels, np.flatnonzero(ok) + 1, stats[ok, :4], cents[ok]


def bright_blobs(gray: np.ndarray, min_area: int = 200) -> np.ndarray:
    return _blobs(gray, min_area)[3]


def dark_dot(gray: np.ndarray, labels: np.ndarray, label: int, box) -> Point | None:
    """Centre of the dark dot inside one note, or None if the note is plain."""
    x, y, w, h = box
    crop = gray[y : y + h, x : x + w]
    # the dot is a hole in the note, so fill the note's hull before looking inside it
    note = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
    inside = np.zeros_like(note)
    cv2.fillConvexPoly(inside, cv2.convexHull(cv2.findNonZero(note)), 1)
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8))
    dark = inside.astype(bool) & (crop < (int(crop.max()) + int(crop.min())) / 2)
    m = cv2.moments(dark.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return None
    return x + m["m10"] / m["m00"], y + m["m01"] / m["m00"]


def pick_pair(before, after, min_move: float = 2.0, spacing: tuple[float, float] = (40, 250)):
    # the notes on the horn are the blobs that moved. the breadboard shifts a bit when
    # its leds blink and a bent note shows as two blobs, so only pair blobs a horn apart
    if len(before) == 0 or len(after) < 2:
        return None
    moved = np.linalg.norm(after[:, None] - before[None], axis=2).min(1)
    best = None
    for i, j in itertools.combinations(np.flatnonzero(moved >= min_move), 2):
        if spacing[0] <= math.dist(after[i], after[j]) <= spacing[1]:
            if best is None or moved[i] + moved[j] > moved[best[0]] + moved[best[1]]:
                best = (i, j)
    return None if best is None else tuple(map(tuple, after[list(best)]))


def track_pair(gray: np.ndarray, pair, prev: float | None = None, max_jump: float | None = None):
    if max_jump is None:
        max_jump = 0.6 * math.dist(*pair)
    labels, ids, boxes, blobs = _blobs(gray)
    pts = []
    for p in pair:
        if len(blobs) == 0:
            return None
        d = np.linalg.norm(blobs - p, axis=1)
        if d.min() > max_jump:
            return None
        j = d.argmin()
        pts.append(dark_dot(gray, labels, ids[j], boxes[j]) or tuple(blobs[j]))
    (x1, y1), (x2, y2) = pts
    if pts[0] == pts[1]:
        return None
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if prev is not None:
        angle = unwrap(angle, prev)
    return angle, pts[0], pts[1]


class PairTracker:
    """Picks the horn's notes by what moved during the find nudge, then follows them.
    Rows are (t, angle_deg)."""

    def __init__(self, top: int):
        self.top = top
        self.first = self.pair = self.prev = None

    def update(self, gray: np.ndarray, t: float, us: int):
        if self.first is None:
            self.first = gray
        elif self.pair is None:
            if us >= self.top - 10:
                self.pair = pick_pair(bright_blobs(self.first), bright_blobs(gray))
        else:
            found = track_pair(gray, self.pair, self.prev)
            if found:
                self.prev, *self.pair = found
                return (t, self.prev)
        return None


def report(commands, angles, rig: dict, axis: int = 0) -> dict:
    """Bare-horn response: commands are (t_send, t_ack, x, y, phase), angles (t, deg)."""
    tc = np.array([c[1] for c in commands], float)
    p = np.array([c[2 + axis] for c in commands], float)
    phase = np.array([c[4] for c in commands])
    ta = np.array([f[0] for f in angles], float)
    a = np.array([f[1] for f in angles], float)

    # second half of the hold, the first half may still be settling from the find nudge
    hold = a[phase_mask(ta, tc, phase, "hold")]
    hold = hold[len(hold) // 2 :]
    noise = max(float(hold.std()), 0.02)
    sw = phase_mask(ta, tc, phase, "sweep")
    k, b = np.polyfit(np.interp(ta[sw], tc, p), a[sw], 1)
    cx = center(rig)[axis]

    steps = [s for s, _ in responses(ta, a, tc, p, phase, "step", noise) if s]
    speeds = [s["speed"] for s in steps if s["speed"] is not None]

    # each rev block moves +d, -2d, +d; the third move is the reversal that must show
    dead = None
    rev_delays = []
    for d in sorted({int(n[3:]) for n in phase if n.startswith("rev")}):
        moves = [s for s, _ in responses(ta, a, tc, p, phase, f"rev{d}", noise)]
        rev_delays += [m["delay"] for m in moves[1:] if m]
        if dead is None and len(moves) == 3 and moves[2]:
            dead = d

    delay_ms = 1000 * float(np.median([s["delay"] for s in steps])) if steps else None
    rev_lag_ms = 1000 * float(np.median(rev_delays)) if rev_delays else None
    reject = dead is None or dead > 25 or (delay_ms and rev_lag_ms and rev_lag_ms > 2 * delay_ms)
    return {
        "us_per_deg": 1 / k,
        "hysteresis_deg": float(hold.mean() - (k * cx + b)),
        "noise_deg": noise,
        "noise_band_us": 3 * noise / abs(k),
        "delay_ms": delay_ms,
        "top_speed_deg_s": float(max(speeds)) if speeds else None,
        "dead_band_us": dead,
        "reversal_lag_ms": rev_lag_ms,
        "verdict": "reject" if reject else "ok",
    }
