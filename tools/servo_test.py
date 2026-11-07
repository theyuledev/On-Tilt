import csv
import sys
import time
from pathlib import Path

import cv2

from src.control.pulse_map import center
from src.io.camera import CameraSource, open_camera
from src.io.serial_link import open_board
from src.io.settings import load
from src.safety import limits
from src.vision.pointer import horn_angle, motion_roi, report
from tools.bench import Bench

REVERSALS = (3, 5, 8, 12, 20, 30)
OUT = Path("data/servo")


def plan(rig: dict, axis: int, tick: float) -> list[tuple[int, str]]:
    """One (pulse_us, phase) per tick: hold, slow sweep, big steps, small reversals."""
    c = round(center(rig)[axis])
    lo, hi = limits(rig)[2 * axis : 2 * axis + 2]
    r = round(min(c - lo, hi - c))
    rate = rig["slew_us_s"] / 4
    out: list[tuple[int, str]] = []

    def hold(us, s, phase):
        out.extend([(us, phase)] * round(s / tick))

    def ramp(a, b, phase):
        n = max(1, round(abs(b - a) / (rate * tick)))
        out.extend((round(a + (b - a) * i / n), phase) for i in range(1, n + 1))

    hold(c, 1.5, "hold")
    ramp(c, c + r, "sweep")
    ramp(c + r, c - r, "sweep")
    ramp(c - r, c, "sweep")
    s = min(100, r)
    for d in (s, 0, -s, 0):
        hold(c + d, 1.0, "step")
    for d in REVERSALS:
        if d > r:
            break
        for k in (d, -d, 0):
            hold(c + k, 0.6, f"rev{d}")
    return out


def run(bench: Bench, cam, steps, other: int, axis: int, tick: float):
    commands, angles = [], []
    first = roi = prev = None
    top = max(us for us, _ in steps)
    last_phase = None
    t0 = time.monotonic()
    for i, (us, phase) in enumerate(steps):
        if phase != last_phase:
            print(f"{time.monotonic() - t0:6.1f}s {phase}", flush=True)
            last_phase = phase
        x, y = (us, other) if axis == 0 else (other, us)
        bench.submit(f"P {x} {y}")
        if bench.last_p is None:
            raise RuntimeError("\n".join(bench.log[-3:]))
        commands.append((time.monotonic(), x, y, phase))
        deadline = t0 + (i + 1) * tick
        while time.monotonic() < deadline:
            f = cam.read()
            if f is None:
                time.sleep(0.001)
                continue
            gray = f.image if f.image.ndim == 2 else cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
            if first is None:
                first = gray
            elif roi is None and us == top:
                roi = motion_roi(first, gray)
                print(f"roi {roi}", flush=True)
            found = horn_angle(gray, roi, prev)
            if found:
                prev = found[0]
                angles.append((f.t_s, prev))
    return commands, angles


def save(name: str, commands, angles) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows(("cmd", *c) for c in commands)
        w.writerows(("ang", *a) for a in angles)
    return path


def main() -> None:
    name = sys.argv[1]
    axis = 1 if sys.argv[2:3] == ["y"] else 0
    cfg = load()
    rig = cfg["rig"]
    tick = 1 / cfg["control"]["sim_tick_hz"]
    steps = plan(rig, axis, tick)
    other = round(center(rig)[1 - axis])
    print(f"{len(steps) * tick:.0f}s planned, ctrl-c kills", flush=True)

    cam = CameraSource(open_camera(rig))
    bench = Bench(open_board(rig), rig)
    bench.submit("l")
    bench.submit("A")
    commands = angles = None
    try:
        commands, angles = run(bench, cam, steps, other, axis, tick)
        bench.submit("D")
    except KeyboardInterrupt:
        bench.kill()
        while bench.killing:
            time.sleep(tick)
            bench.tick(tick)
    finally:
        cam.close()
    if not commands:
        return
    print(f"wrote {save(name, commands, angles)} ({len(angles)} frames)", flush=True)
    if len(angles) < 10:
        print("too few angles, pointer not found")
        return
    for k, v in report(commands, angles, rig, axis).items():
        print(f"{k:>18}: {v if not isinstance(v, float) else round(v, 2)}")


if __name__ == "__main__":
    main()
