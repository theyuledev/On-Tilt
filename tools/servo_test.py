import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.control.pulse_map import center
from src.io.camera import CameraSource, open_camera
from src.io.serial_link import open_board
from src.io.settings import ROOT, load
from src.safety import limits
from src.vision.calib import load_calib
from src.vision.markers import TiltTracker, load_board
from src.vision.plate_pose import tilt_report
from src.vision.pointer import PairTracker, report
from tools.bench import Bench, kill_and_wait

REVERSALS = (3, 5, 8, 12, 20, 30)
OUT = ROOT / "data" / "servo"
BOARD = ROOT / "configs" / "board.local.json"


def plan(rig: dict, axis: int, tick: float) -> list[tuple[int, str]]:
    """One (pulse_us, phase) per tick: find nudge, hold, slow sweep, big steps, small reversals."""
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

    ramp(c, c + r, "find")
    ramp(c + r, c, "find")
    hold(c, 1.5, "hold")
    ramp(c, c + r, "sweep")
    ramp(c + r, c - r, "sweep")
    ramp(c - r, c, "sweep")
    s = min(100, r)
    for d in (0, s, 0, -s, 0):
        hold(c + d, 1.0, "step")
    for d in REVERSALS:
        if d > r:
            break
        for k in (d, -d, 0):
            hold(c + k, 0.6, f"rev{d}")
    return out


def run(bench, cam, steps, other, axis, tick, tracker, clock=None, commands=None, rows=None):
    """Commands are (t_send, t_ack, x, y, phase), rows what the tracker returns.
    Pass the lists in to keep what was logged if the run is cut short."""
    clock = clock or time.monotonic
    commands = [] if commands is None else commands
    rows = [] if rows is None else rows
    last_phase = None
    t0 = clock()
    for i, (us, phase) in enumerate(steps):
        if phase != last_phase:
            print(f"{clock() - t0:6.1f}s {phase}", flush=True)
            last_phase = phase
        x, y = (us, other) if axis == 0 else (other, us)
        t_send = clock()
        reply = bench.submit(f"P {x} {y}")
        if not reply.startswith("OK P"):
            raise RuntimeError(reply or "\n".join(bench.log[-3:]))
        commands.append((t_send, clock(), x, y, phase))
        deadline = t0 + (i + 1) * tick
        while clock() < deadline:
            f = cam.read()
            if f is None:
                time.sleep(0.001)
                continue
            gray = f.image if f.image.ndim == 2 else cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
            row = tracker.update(gray, f.t_s, us)
            if row:
                rows.append(row)
    return commands, rows


def guarded(bench: Bench, tick: float, fn):
    """fn() with the servos ramped to level and disarmed on any error; None on ctrl-c."""
    try:
        out = fn()
    except BaseException as e:
        kill_and_wait(bench, tick)
        if isinstance(e, KeyboardInterrupt):
            return None
        raise
    bench.submit("D")
    return out


def save(name: str, commands, rows, kind: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerows(("cmd", *c) for c in commands)
        w.writerows((kind, *r) for r in rows)
    return path


def load_csv(path: Path):
    """(commands, rows, kind); older files have one command time, used for both."""
    commands, rows, kind = [], [], "ang"
    for row in csv.reader(path.open()):
        if row[0] == "cmd":
            t = [float(v) for v in row[1:-3]]
            commands.append((t[0], t[-1], int(row[-3]), int(row[-2]), row[-1]))
        else:
            kind = row[0]
            rows.append(tuple(float(v) for v in row[1:]))
    return commands, rows, kind


def show(commands, rows, kind: str, rig: dict) -> None:
    axis = 0 if len({c[2] for c in commands}) > 1 else 1
    out = tilt_report(commands, rows, axis) if kind == "tilt" else report(commands, rows, rig, axis)
    for k, v in out.items():
        print(f"{k:>18}: {v if not isinstance(v, float) else round(v, 3)}")


def annotate(tracker: TiltTracker, path: Path) -> None:
    """The last frame the plate was solved on, markers outlined with their ids."""
    img = tracker.image
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    ids = np.array(list(tracker.found), np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedMarkers(img, [c.reshape(1, 4, 2) for c in tracker.found.values()], ids)
    cv2.imwrite(str(path), img)


def annotate_pair(tracker: PairTracker, path: Path) -> None:
    """The frame the notes were picked on, both circled, so you can see what got tracked."""
    (x1, y1), (x2, y2) = (tuple(map(round, p)) for p in tracker.seed)
    img = cv2.cvtColor(tracker.image, cv2.COLOR_GRAY2BGR)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for p in ((x1, y1), (x2, y2)):
        cv2.circle(img, p, 25, (0, 0, 255), 2)
    cv2.imwrite(str(path), img)
    span = math.dist((x1, y1), (x2, y2))
    print(f"pair at ({x1},{y1})/({x2},{y2}), {span:.0f} px apart, see {path}", flush=True)


def main() -> None:
    cfg = load()
    rig = cfg["rig"]
    if sys.argv[1] == "--report":
        show(*load_csv(Path(sys.argv[2])), rig)
        return
    name, args = sys.argv[1], sys.argv[2:]
    axis = 1 if "y" in args else 0
    tick = 1 / cfg["control"]["sim_tick_hz"]
    steps = plan(rig, axis, tick)
    other = round(center(rig)[1 - axis])
    if "plate" in args:
        board = next((Path(a) for a in args if a.endswith(".json")), BOARD)
        tracker = TiltTracker(cfg["vision"], load_calib(), *load_board(board))
        name, kind = f"{name}_plate", "tilt"
    else:
        tracker, kind = PairTracker(max(us for us, _ in steps)), "ang"
    print(f"{len(steps) * tick:.0f}s planned, ctrl-c kills", flush=True)
    cam = CameraSource(open_camera(rig))
    bench = Bench(open_board(rig), rig)
    bench.submit("l")
    bench.submit("A")
    commands, rows = [], []
    try:
        done = guarded(
            bench,
            tick,
            lambda: run(bench, cam, steps, other, axis, tick, tracker, None, commands, rows),
        )
    finally:
        cam.close()
    finish(name, commands, rows, tracker, kind, rig, done is None)


def finish(name, commands, rows, tracker, kind: str, rig: dict, cut: bool) -> None:
    if cut:
        print("partial run, ctrl-c")
        if not (commands or rows):
            return
    print(f"wrote {save(name, commands, rows, kind)} ({len(rows)} frames)", flush=True)
    if kind == "ang" and tracker.seed:
        annotate_pair(tracker, OUT / f"{name}.jpg")
    seen = sum(not math.isnan(r[1]) for r in rows)
    if seen < 10 and not cut:
        print("too few frames, target not found")
    if cut or seen < 10:
        return
    if kind == "tilt":
        annotate(tracker, OUT / f"{name}.jpg")
    show(commands, rows, kind, rig)


if __name__ == "__main__":
    main()
