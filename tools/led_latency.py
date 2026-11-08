import csv
import random
import shutil
import sys
import time
from typing import NamedTuple

import numpy as np

from src.io.camera import CameraSource, open_camera
from src.io.serial_link import open_board
from src.io.settings import ROOT, load
from tools import led_math
from tools.led_math import MIN_CONTRAST, contrast_map, level, pick_window, summarize

OUT = ROOT / "data" / "latency"
SETTLE_S = 0.3
POLL_S = 0.0005
WAIT_S = 0.5
FRAME_TIMEOUT_S = 1.0


class Hit(NamedTuple):
    t_cmd: float
    t_ack: float
    t_lit: float
    frames_to_lit: int
    level_dark: float
    level_lit: float


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
        if f is not None:
            out.append(f)
        elif clock() > deadline:
            raise RuntimeError("camera stopped sending frames")
        else:
            sleep(POLL_S)
    return out


def average(cam, clock, sleep):
    return np.mean([f.image for f in frames(cam, 10, clock, sleep)], axis=0, dtype=np.float32)


def find_led(link, cam, clock, sleep, roi=None):
    """(window, dark_below, lit_above) or None when the led is not in the picture."""
    marker(link, False)
    sleep(SETTLE_S)
    off = average(cam, clock, sleep)
    marker(link, True)
    sleep(SETTLE_S)
    on = average(cam, clock, sleep)
    window = roi or pick_window(off, on)
    if window is None:
        print(f"led not in view, contrast {contrast_map(off, on).max():.0f} < {MIN_CONTRAST}")
        return None
    lo, hi = level(off, window), level(on, window)
    if warning := led_math.saturation_warning(hi):
        print(warning)
    return window, lo + (hi - lo) / 4, (lo + hi) / 2


def trial(link, cam, led, clock, sleep, rand) -> Hit | None:
    window, dark, lit = led
    marker(link, False)
    sleep(SETTLE_S)
    level_dark = level(frames(cam, 1, clock, sleep)[0].image, window)
    if not level_dark < dark:
        return None
    sleep(rand(0.05, 0.25))
    cam.read()
    t0 = clock()
    marker(link, True)
    t_ack = clock()
    n = 0
    while clock() - t0 < WAIT_S:
        f = cam.read()
        if f is None:
            sleep(POLL_S)
            continue
        n += 1
        lvl = level(f.image, window)
        if f.t_s >= t0 and lvl > lit:
            return Hit(t0, t_ack, f.t_s, n, level_dark, lvl)
    return None


def run(
    link,
    cam,
    trials,
    clock=time.monotonic,
    sleep=time.sleep,
    rand=random.uniform,
    on_hit=None,
    roi=None,
) -> tuple[list[float], list[float], int]:
    """(latencies, rtts, misses); stops early on ctrl-c and keeps what it has."""
    link.flush()
    lat: list[float] = []
    rtt: list[float] = []
    misses = 0
    led = find_led(link, cam, clock, sleep, roi)
    if led is None:
        return lat, rtt, misses
    try:
        for _ in range(trials):
            hit = trial(link, cam, led, clock, sleep, rand)
            if hit is None:
                misses += 1
                print("miss", flush=True)
                continue
            lat.append(hit.t_lit - hit.t_cmd)
            rtt.append(hit.t_ack - hit.t_cmd)
            if on_hit is not None:
                on_hit(hit)
    except KeyboardInterrupt:
        print("stopped early", flush=True)
    return lat, rtt, misses


def show(out: dict, misses: int, fps: float) -> None:
    for k, v in out.items():
        print(f"{k:>8}: {round(v, 4) if isinstance(v, float) else v}")
    print(f"  misses: {misses}")
    print("l_mid: exposure midpoint to frame timestamp, mean minus half a frame and half the rtt")
    print("l_min: the same from the floor, min minus half the rtt")
    print("sanity check: l_min close to l_mid, spread close to one frame period")
    if out["spread"] > 1.5 / fps:
        print("irregular frame clock: spread is over 1.5 frame periods")
    p95, n = out["p95"], out["n"]
    print(f"camera_latency_s = {out['l_mid']:.4f} (p95 latency {p95:.4f}, n {n}, misses {misses})")


def parse(args: list[str]):
    """(name, trials, roi); --roi x,y,w,h uses that window instead of the auto pick."""
    roi = None
    if "--roi" in args:
        i = args.index("--roi")
        x, y, w, h = (int(v) for v in args[i + 1].split(","))
        roi = (x, x + w, y, y + h)
        del args[i : i + 2]
    return args[0], int(args[1]) if len(args) > 1 else 50, roi


def main() -> None:
    name, trials, roi = parse(sys.argv[1:])
    rig = load()["rig"]
    fps = rig["camera_fps"]
    cam = CameraSource(open_camera(rig))
    print(f"manual exposure via uvc-util: {shutil.which('uvc-util') is not None}", flush=True)
    link = open_board(rig)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("trial", *Hit._fields))
        n = 0

        def on_hit(hit: Hit) -> None:
            nonlocal n
            n += 1
            w.writerow((n, *(f"{v:.6f}" if isinstance(v, float) else v for v in hit)))
            f.flush()
            lat, rtt = hit.t_lit - hit.t_cmd, hit.t_ack - hit.t_cmd
            print(f"{n:3d} {lat * 1000:6.2f} ms rtt {rtt * 1000:.2f} ms {hit.frames_to_lit} fr")

        try:
            lat, rtt, misses = run(link, cam, trials, on_hit=on_hit, roi=roi)
        finally:
            cam.close()
            try:
                marker(link, False)
            except Exception:  # a dead link must not hide the real error
                pass
    if not lat:
        print(f"no timings, misses: {misses}")
        sys.exit(1)
    print(f"wrote {path}")
    show(summarize(lat, rtt, fps), misses, fps)


if __name__ == "__main__":
    main()
