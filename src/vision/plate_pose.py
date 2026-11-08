"""Plate step response from logged tilts, one axis at a time."""

import numpy as np


def square(size_mm: float) -> np.ndarray:
    s = size_mm / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)


def phase_mask(t: np.ndarray, tc: np.ndarray, phase: np.ndarray, name: str) -> np.ndarray:
    """Samples at t that fall inside the commands' phase `name`, none if it never ran."""
    sel = np.flatnonzero(phase == name)
    if len(sel) == 0:
        return np.zeros(len(t), bool)
    end = tc[sel[-1] + 1] if sel[-1] + 1 < len(tc) else np.inf
    return (t >= tc[sel[0]]) & (t < end)


def no_data(t, tc, phase) -> list[str]:
    """Phases with under two samples, e.g. a run cut short before them."""
    return [n for n in ("hold", "sweep", "step") if phase_mask(t, tc, phase, n).sum() < 2]


def sweep_fit(t, y, tc, p, phase):
    """(slope, offset) of y against the pulse over the sweep, None under two samples."""
    sw = phase_mask(t, tc, phase, "sweep")
    return np.polyfit(np.interp(t[sw], tc, p), y[sw], 1) if sw.sum() > 1 else None


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


def _axis(tilts, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame times and one axis of the rows that have a tilt, one row per frame."""
    tt = np.array([r[0] for r in tilts], float)
    a = np.array([r[1 + axis] for r in tilts], float)
    ok = np.isfinite(a)
    tt, first = np.unique(tt[ok], return_index=True)
    return tt, a[ok][first]


def tilt_report(commands, tilts, axis: int) -> dict:
    """Loaded step response of one axis: commands are (t_send, t_ack, x, y, phase),
    tilts are (t, tilt_x, tilt_y, ...), nan where the plate was not seen. hold_deg is
    where the plate settles on the hold pulse, trim_us the pulse change that levels it."""
    tc = np.array([c[1] for c in commands], float)
    p = np.array([c[2 + axis] for c in commands], float)
    phase = np.array([c[4] for c in commands])
    tt, a = _axis(tilts, axis)
    hold = a[phase_mask(tt, tc, phase, "hold")]
    level = float(hold[len(hold) // 2 :].mean()) if len(hold) else None
    noise = max(float(hold[len(hold) // 2 :].std()) if len(hold) else 0, 0.02)
    dt = np.diff(tt)
    med = float(np.median(dt))
    fit = sweep_fit(tt, a, tc, p, phase)
    moved = [(s, dp) for s, dp in responses(tt, a, tc, p, phase, "step", noise) if s]
    steps = [s for s, _ in moved]
    return {
        "delay_ms": _median_ms(steps, "delay"),
        "rise_ms": _median_ms(steps, "rise"),
        "settle_ms": _median_ms(steps, "settle"),
        "overshoot_pct": max(s["overshoot"] for s in steps) if steps else None,
        "deg_per_us": float(np.mean([abs(s["change"] / dp) for s, dp in moved])) if moved else None,
        "deg_per_us_sweep": None if fit is None else float(fit[0]),
        "noise_deg": noise if len(hold) else None,
        "hold_deg": level,
        "trim_us": None if level is None or fit is None else -level / float(fit[0]),
        "fps": 1 / med,
        "dropped_frames": int((dt > 1.5 * med).sum()),
        "frames": len(tt),
        "no_data": no_data(tt, tc, phase),
    }
