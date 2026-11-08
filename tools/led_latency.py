import csv
import random
import shutil
import statistics
import sys
import time

import cv2
import numpy as np

from src.io.camera import CameraSource, open_camera
from src.io.serial_link import LinkError, open_board
from src.io.settings import ROOT, load

OUT = ROOT / "data" / "latency"
MIN_CONTRAST = 30  # gray levels, led on minus off
SETTLE_S = 0.3
POLL_S = 0.0005
WAIT_S = 0.5
FRAME_TIMEOUT_S = 1.0


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


def summarize(lat: list[float], rtt: list[float], fps: float, exposure_s: float) -> dict:
    mean_rtt = statistics.fmean(rtt)
    out = {
        "n": len(lat),
        "mean": statistics.fmean(lat),
        "median": statistics.median(lat),
        "min": min(lat),
        "max": max(lat),
        "std": statistics.stdev(lat) if len(lat) > 1 else 0.0,
        "mean_rtt": mean_rtt,
    }
    out["l_mid"] = out["mean"] - 0.5 / fps - exposure_s / 2 - mean_rtt / 2
    out["l_min"] = out["min"] - mean_rtt / 2
    out["spread"] = out["max"] - out["min"]
    return out


def marker(link, on: bool) -> None:
    line = f"M {int(on)}"
    reply = link.raw(line)
    if reply != "OK M":
        raise RuntimeError(f"{line} -> {reply!r}")


def frames(cam, n: int, clock, sleep) -> list:
    out: list = []
    deadline = clock() + FRAME_TIMEOUT_S
    while len(out) < n:
        f = cam.read()
        if f is None:
            if clock() > deadline:
                raise RuntimeError("camera stopped sending frames")
            sleep(POLL_S)
            continue
        out.append(f)
    return out


def average(cam, clock, sleep):
    return np.mean([f.image for f in frames(cam, 10, clock, sleep)], axis=0, dtype=np.float32)


def find_led(link, cam, clock, sleep):
    """(window, dark_below, lit_above) or None when the led is not in the picture."""
    marker(link, False)
    sleep(SETTLE_S)
    off = average(cam, clock, sleep)
    marker(link, True)
    sleep(SETTLE_S)
    on = average(cam, clock, sleep)
    window = pick_window(off, on)
    if window is None:
        print(f"led not in view, contrast {contrast_map(off, on).max():.0f} < {MIN_CONTRAST}")
        return None
    lo, hi = level(off, window), level(on, window)
    return window, lo + (hi - lo) / 4, (lo + hi) / 2


def trial(link, cam, led, clock, sleep, rand):
    """(latency, rtt) or None for a miss."""
    window, dark, lit = led
    marker(link, False)
    sleep(SETTLE_S)
    if not level(frames(cam, 1, clock, sleep)[0].image, window) < dark:
        return None
    sleep(rand(0.05, 0.25))
    cam.read()
    t0 = clock()
    marker(link, True)
    t_ack = clock()
    while clock() - t0 < WAIT_S:
        f = cam.read()
        if f is None:
            sleep(POLL_S)
        elif f.t_s >= t0 and level(f.image, window) > lit:
            return f.t_s - t0, t_ack - t0
    return None


def run(
    link, cam, trials, fps, clock=time.monotonic, sleep=time.sleep, rand=random.uniform, on_row=None
) -> tuple[list[float], list[float], int]:
    """(latencies, rtts, misses); stops early on ctrl-c and keeps what it has."""
    link.flush()
    lat: list[float] = []
    rtt: list[float] = []
    misses = 0
    led = find_led(link, cam, clock, sleep)
    if led is None:
        return lat, rtt, misses
    try:
        for _ in range(trials):
            got = trial(link, cam, led, clock, sleep, rand)
            if got is None:
                misses += 1
                print("miss", flush=True)
                continue
            lat.append(got[0])
            rtt.append(got[1])
            if on_row is not None:
                on_row(*got)
    except KeyboardInterrupt:
        print("stopped early", flush=True)
    return lat, rtt, misses


def show(out: dict, misses: int, fps: float) -> None:
    for k, v in out.items():
        print(f"{k:>8}: {round(v, 4) if isinstance(v, float) else v}")
    print(f"  misses: {misses}")
    print("l_mid: exposure midpoint to frame timestamp (frame t_s minus l_mid = scene time)")
    print("l_min: floor, min minus half rtt")
    if out["spread"] > 1.5 / fps:
        print("irregular frame clock: spread is over 1.5 frame periods")


def main() -> None:
    name = sys.argv[1]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    rig = load()["rig"]
    fps, exposure_s = rig["camera_fps"], rig["camera_exposure_ms"] / 1000
    cam = CameraSource(open_camera(rig))
    print(f"manual exposure via uvc-util: {shutil.which('uvc-util') is not None}", flush=True)
    link = open_board(rig)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("trial", "latency_s", "rtt_s"))
        n = 0

        def on_row(latency, rtt):
            nonlocal n
            n += 1
            w.writerow((n, f"{latency:.6f}", f"{rtt:.6f}"))
            f.flush()
            print(f"{n:3d} {latency * 1000:6.2f} ms  rtt {rtt * 1000:.2f} ms", flush=True)

        try:
            lat, rtt, misses = run(link, cam, trials, fps, on_row=on_row)
        finally:
            try:
                marker(link, False)
            except LinkError:
                pass
            cam.close()
    if not lat:
        print(f"no timings, misses: {misses}")
        sys.exit(1)
    print(f"wrote {path}")
    show(summarize(lat, rtt, fps, exposure_s), misses, fps)


if __name__ == "__main__":
    main()
