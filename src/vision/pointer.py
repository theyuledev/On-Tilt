import itertools
import math

import cv2
import numpy as np

from src.control.pulse_map import center
from src.vision.plate_pose import no_data, phase_mask, responses, sweep_fit

Point = tuple[float, float]


def unwrap(angle: float, prev: float) -> float:
    # the two pointers are interchangeable so the line is only known mod 180
    while angle - prev > 90:
        angle -= 180
    while angle - prev < -90:
        angle += 180
    return angle


def bright_level(gray: np.ndarray) -> float:
    return (np.percentile(gray, 50) + np.percentile(gray, 99.9)) / 2


def moved_pair(first, gray, spacing: tuple[float, float] = (40, 250), min_area: int = 50):
    """The two biggest bright patches a horn apart that moved since the rest frame."""
    # the breadboard shifts a px or two when its leds blink and a bent note shows as two
    # patches, so pair by span and take the most area, no shape tests
    diff = cv2.GaussianBlur(cv2.absdiff(first, gray), (7, 7), 0)
    noise = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    moved = (diff > max(20, 5 * noise)) & (gray > bright_level(gray))
    n, _, stats, cents = cv2.connectedComponentsWithStats(moved.astype(np.uint8))
    area = stats[:, cv2.CC_STAT_AREA]
    ids = [i for i in range(1, n) if area[i] >= min_area]
    best = None
    for i, j in itertools.combinations(ids, 2):
        if spacing[0] <= math.dist(cents[i], cents[j]) <= spacing[1]:
            if best is None or area[i] + area[j] > area[best[0]] + area[best[1]]:
                best = (i, j)
    return None if best is None else (tuple(cents[best[0]]), tuple(cents[best[1]]))


def dark_dot(crop: np.ndarray, note: np.ndarray) -> Point | None:
    """Centre of the dark dot inside one note, or None if the note is plain."""
    # the dot is a hole in the note, so fill the note's hull before looking inside it
    inside = np.zeros_like(note)
    cv2.fillConvexPoly(inside, cv2.convexHull(cv2.findNonZero(note)), 1)
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8))
    dark = inside.astype(bool) & (crop < (int(crop.max()) + int(crop.min())) / 2)
    m = cv2.moments(dark.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def note_centre(gray: np.ndarray, p: Point, h: int, thr: float) -> Point | None:
    """Mean-shift centroid of the bright pixels within h px of p, on the note's dot if any."""
    for _ in range(2):
        x0, y0 = max(0, int(round(p[0])) - h), max(0, int(round(p[1])) - h)
        crop = gray[y0 : y0 + 2 * h, x0 : x0 + 2 * h]
        # the biggest bright patch in the window, the other note may poke into it
        n, labels, stats, cents = cv2.connectedComponentsWithStats((crop > thr).astype(np.uint8))
        if n < 2:
            return None
        k = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        if stats[k, cv2.CC_STAT_AREA] < 30:
            return None
        note = (labels == k).astype(np.uint8)
        p = (x0 + cents[k, 0], y0 + cents[k, 1])
    dot = dark_dot(crop, note)
    return p if dot is None else (x0 + dot[0], y0 + dot[1])


class PairTracker:
    """Picks the horn's notes by what moved during the find nudge, then follows each with a
    window centroid. Rows are (t, angle_deg)."""

    def __init__(self, top: int):
        self.top = top
        self.first = self.image = self.seed = self.pair = self.prev = self.thr = None

    def update(self, gray: np.ndarray, t: float, us: int):
        if self.first is None:
            self.first = gray
        elif self.pair is None:
            if us >= self.top - 10:
                self.pair = moved_pair(self.first, gray)
                if self.pair:
                    # the bright level is frozen here so a frame without notes stays empty
                    self.seed, self.image, self.thr = self.pair, gray, bright_level(gray)
        else:
            h = int(np.clip(0.35 * math.dist(*self.pair), 20, 60))
            pts = [note_centre(gray, p, h, self.thr) for p in self.pair]
            if None not in pts:
                self.pair = pts
                (x1, y1), (x2, y2) = pts
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                self.prev = angle if self.prev is None else unwrap(angle, self.prev)
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
    noise = max(float(hold.std()) if len(hold) else 0, 0.02)
    fit = sweep_fit(ta, a, tc, p, phase)
    k, b = (None, None) if fit is None else fit
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
        "us_per_deg": None if fit is None else 1 / k,
        "hysteresis_deg": None if fit is None else float(hold.mean() - (k * cx + b)),
        "noise_deg": noise if len(hold) else None,
        "noise_band_us": None if fit is None else 3 * noise / abs(k),
        "delay_ms": delay_ms,
        "top_speed_deg_s": float(max(speeds)) if speeds else None,
        "dead_band_us": dead,
        "reversal_lag_ms": rev_lag_ms,
        "verdict": "reject" if reject else "ok",
        "no_data": no_data(ta, tc, phase),
    }
