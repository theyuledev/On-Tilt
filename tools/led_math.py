import statistics

import cv2
import numpy as np

MIN_CONTRAST = 30  # gray levels, led on minus off


def level(image, window) -> float:
    x0, x1, y0, y1 = window
    return float(image[y0:y1, x0:x1].mean())


def contrast_map(off, on):
    if off.ndim == 3:
        off, on = off.mean(axis=2), on.mean(axis=2)
    return cv2.GaussianBlur((on - off).astype(np.float32), (5, 5), 0)


def pick_window(off, on, half=4) -> tuple[int, int, int, int] | None:
    """(x0, x1, y0, y1) around the brightest change, None when the led does not show."""
    diff = contrast_map(off, on)
    if diff.max() < MIN_CONTRAST:
        return None
    y, x = (int(v) for v in np.unravel_index(diff.argmax(), diff.shape))
    h, w = diff.shape
    return max(0, x - half), min(w, x + half + 1), max(0, y - half), min(h, y + half + 1)


def summarize(lat: list[float], rtt: list[float], fps: float) -> dict:
    """No exposure term: the first lit frame's midpoint averages half a period past the led."""
    mean_rtt = statistics.fmean(rtt)
    qs = statistics.quantiles(lat, n=20, method="inclusive") if len(lat) > 1 else lat * 19
    out = {
        "n": len(lat),
        "mean": statistics.fmean(lat),
        "median": statistics.median(lat),
        "p50": qs[9],
        "p95": qs[18],
        "min": min(lat),
        "max": max(lat),
        "std": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "mean_rtt": mean_rtt,
    }
    out["l_mid"] = out["mean"] - 0.5 / fps - mean_rtt / 2
    out["l_min"] = out["min"] - mean_rtt / 2
    out["spread"] = out["max"] - out["min"]
    return out
