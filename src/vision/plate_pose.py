"""Plate tilt from the bright quad the overhead cam sees.

Corner 0 is the top-most corner in the image, 1..3 follow clockwise. The plate's
+x edge runs corner 0 -> 1 and +y runs corner 3 -> 0, so for the diamond the cam
sees today +x points down-right and +y up-right in the image. A tilt is positive
when that + edge drops away from the camera. The image-to-plate axis mapping is
still to be pinned down on the rig.
"""

import cv2
import numpy as np

from src.vision.calib import undistort

SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 0.01)


def _order(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts[:, 1])), axis=0)


def _match(quad: np.ndarray, prev: np.ndarray) -> np.ndarray:
    k = min(range(4), key=lambda k: np.linalg.norm(np.roll(quad, -k, 0) - prev, axis=1).sum())
    return np.roll(quad, -k, axis=0)


def find_plate_quad(gray: np.ndarray, prev: np.ndarray | None = None) -> np.ndarray | None:
    """Corners of the biggest bright quad as float32 (4, 2), or None."""
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = gray.shape[0] * gray.shape[1]
    big = [c for c in contours if 0.02 * area < cv2.contourArea(c) < 0.9 * area]
    if not big:
        return None
    hull = cv2.convexHull(max(big, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    poly = hull
    for eps in np.arange(0.01, 0.2, 0.01):
        poly = cv2.approxPolyDP(hull, eps * peri, True)
        if len(poly) <= 4:
            break
    if len(poly) != 4:
        poly = cv2.boxPoints(cv2.minAreaRect(hull))
    pts = np.ascontiguousarray(poly.reshape(4, 2), np.float32)
    cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), SUBPIX)
    quad = _order(pts)
    return _match(quad, prev) if prev is not None else quad


def square(size_mm: float) -> np.ndarray:
    s = size_mm / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)


def plate_tilt(corners: np.ndarray, lens: dict, size_mm: float):
    """(tilt_x_deg, tilt_y_deg, rvec, tvec) of a size_mm square seen at these corners
    through the calib file's lens, standard or fisheye."""
    pts = undistort(lens, corners).reshape(-1, 1, 2)
    eye = np.eye(3)
    ok, rvec, tvec = cv2.solvePnP(square(size_mm), pts, eye, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    r, _ = cv2.Rodrigues(rvec)
    # camera z points down at the plate, so an edge with +z is the low one
    tilt = np.degrees(np.arcsin(np.clip(r[2, :2], -1, 1)))
    return float(tilt[0]), float(tilt[1]), rvec, tvec


class QuadTracker:
    """Follows the plate quad frame to frame; rows are (t, tilt_x, tilt_y, x0, y0, .., y3)."""

    def __init__(self, lens: dict, size_mm: float):
        self.lens, self.size_mm = lens, size_mm
        self.quad = self.image = None

    def update(self, gray: np.ndarray, t: float, us: int | None = None):
        quad = find_plate_quad(gray, self.quad)
        if quad is None:
            return None
        self.quad, self.image = quad, gray
        tilt = plate_tilt(quad, self.lens, self.size_mm)
        return None if tilt is None else (t, tilt[0], tilt[1], *map(float, quad.ravel()))


def phase_mask(t: np.ndarray, tc: np.ndarray, phase: np.ndarray, name: str) -> np.ndarray:
    """Samples at t that fall inside the commands' phase `name`."""
    sel = np.flatnonzero(phase == name)
    end = tc[sel[-1] + 1] if sel[-1] + 1 < len(tc) else np.inf
    return (t >= tc[sel[0]]) & (t < end)


def _first(cond: np.ndarray) -> int | None:
    hit = np.flatnonzero(cond)
    return int(hit[0]) if len(hit) else None


def step_response(t, y, t0: float, t1: float, noise: float) -> dict | None:
    """Delay, 10-90 rise, 5 % settle (s), overshoot %, 20-80 % slope and final change
    of y for a step commanded at t0, or None if y never left the noise band."""
    base = y[(t > t0 - 0.1) & (t <= t0)]
    win = (t > t0) & (t <= t1)
    if len(base) == 0 or not win.any():
        return None
    tw, dev = t[win], y[win] - base.mean()
    final = dev[-max(len(dev) // 3, 1) :].mean()
    if abs(final) < 3 * noise:
        return None
    d, f = dev * np.sign(final), abs(final)
    # sparse rows can pass the final test with no sample actually crossing
    levels = (3 * noise, 0.1 * f, 0.2 * f, 0.8 * f, 0.9 * f)
    i3, i10, i20, i80, i90 = (_first(d > k) for k in levels)
    if None in (i3, i10, i20, i80, i90):
        return None
    out = np.flatnonzero(np.abs(d - f) > max(0.05 * f, 3 * noise))
    settled = out[-1] + 1 if len(out) else 0
    return {
        "delay": tw[i3] - t0,
        "rise": tw[i90] - tw[i10],
        "settle": tw[settled] - t0 if settled < len(tw) else None,
        "overshoot": max(0.0, (d.max() - f) / f * 100),
        "speed": np.polyfit(tw[i20 : i80 + 1], d[i20 : i80 + 1], 1)[0] if i80 > i20 else None,
        "change": final,
    }


def responses(t, y, tc, p, phase, name: str, noise: float) -> list[tuple[dict | None, float]]:
    """(step_response, pulse change) for every pulse change inside phase `name`."""
    moves = np.flatnonzero(p[1:] != p[:-1]) + 1
    if len(moves) == 0:
        return []
    ends = np.append(tc[moves[1:]], t[-1])
    return [
        (step_response(t, y, tc[i], t1, noise), p[i] - p[i - 1])
        for i, t1 in zip(moves, ends, strict=True)
        if phase[i] == name
    ]


def _median_ms(steps: list[dict], key: str) -> float | None:
    vals = [s[key] for s in steps if s[key] is not None]
    return 1000 * float(np.median(vals)) if vals else None


def tilt_report(commands, tilts, axis: int) -> dict:
    """Loaded step response of one axis: commands are (t_send, t_ack, x, y, phase),
    tilts are (t, tilt_x, tilt_y, ...)."""
    tc = np.array([c[1] for c in commands], float)
    p = np.array([c[2 + axis] for c in commands], float)
    phase = np.array([c[4] for c in commands])
    tt = np.array([r[0] for r in tilts], float)
    a = np.array([r[1 + axis] for r in tilts], float)
    hold = a[phase_mask(tt, tc, phase, "hold")]
    noise = max(float(hold[len(hold) // 2 :].std()), 0.02)
    dt = np.diff(tt)
    med = float(np.median(dt))
    sw = phase_mask(tt, tc, phase, "sweep")
    k = np.polyfit(np.interp(tt[sw], tc, p), a[sw], 1)[0]
    moved = [(s, dp) for s, dp in responses(tt, a, tc, p, phase, "step", noise) if s]
    steps = [s for s, _ in moved]
    return {
        "delay_ms": _median_ms(steps, "delay"),
        "rise_ms": _median_ms(steps, "rise"),
        "settle_ms": _median_ms(steps, "settle"),
        "overshoot_pct": max(s["overshoot"] for s in steps) if steps else None,
        "deg_per_us": float(np.mean([abs(s["change"] / dp) for s, dp in moved])) if moved else None,
        "deg_per_us_sweep": float(k),
        "noise_deg": noise,
        "fps": 1 / med,
        "dropped_frames": int((dt > 1.5 * med).sum()),
        "frames": len(tt),
    }
