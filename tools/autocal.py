import argparse
import datetime
import sys
import time

import numpy as np

from src.control.autocal import Autocal
from src.control.pulse_map import center
from src.io.settings import load
from tools.bench import Bench, kill_and_wait


class Abort(RuntimeError):
    pass


def still(cam, clock, sleep, secs: float, period: float) -> list:
    rows, end = [], clock() + secs
    while clock() < end:
        obs = cam.latest()
        if obs is not None and obs.ball_px.valid and (not rows or obs.t_read > rows[-1].t_s):
            rows.append(obs.ball_px)
        sleep(period)
    return rows


def refused(bench: Bench, start: int) -> str | None:
    for entry in bench.log[start:]:
        line, _, reply = entry.partition("  ->  ")
        if reply and not reply.startswith("OK"):
            return entry
    return None


def centre_and_scale(cam, cfg, clock, sleep, ask, period):
    c = cfg["control"]
    while True:
        ask("ball on the centre mark, then Enter ")
        rows = still(cam, clock, sleep, c["autocal_still_s"], period)
        if len(rows) < c["autocal_warm_obs"]:
            raise Abort(f"only {len(rows)} ball sightings, is the camera on")
        x, y, r = (np.array([getattr(p, k) for p in rows]) for k in ("x_px", "y_px", "r_px"))
        if max(x.std(), y.std()) < c["autocal_centre_spread_px"]:
            break
        print(f"ball moved ({x.std():.1f}, {y.std():.1f} px), again")
    r_px = float(np.median(r)) + c["autocal_r_short_px"]
    return (float(np.median(x)), float(np.median(y))), r_px / (cfg["rig"]["ball_d_mm"] / 2)


def pulses(bench: Bench, cam, cal: Autocal, clock, sleep, period) -> None:
    start = len(bench.log)
    if bench.submit("A") != "OK A":
        raise Abort("board did not arm")
    last = clock()
    while True:
        now = clock()
        cmd = cal.step(now, cam.latest())
        if cmd is None:
            break
        if bench.last_p != (cmd.x_us, cmd.y_us):
            bench.submit(f"P {cmd.x_us} {cmd.y_us}")
        else:
            bench.tick(now - last)
        last = now
        bad = refused(bench, start)
        if bad:
            raise Abort(bad)
        sleep(period)
    if cal.aborted:
        raise Abort(cal.aborted)


def report(cal, centre, s_ball: float, cfg: dict) -> str:
    r, rig = cal.result, cfg["rig"]
    tag = f'label = "measured ({datetime.date.today().isoformat()})" }}'
    old = cfg["vision"]["px_per_mm"]
    half = rig["ball_d_mm"] / 2
    h = f"{half * s_ball / (s_ball - old):.0f} mm" if s_ball > old else "n/a"
    lines = [
        "[rig]",
        f"trim_x_us = {{ value = {round(r.trim_x_us)}, {tag}",
        f"trim_y_us = {{ value = {round(r.trim_y_us)}, {tag}",
        f"us_per_deg_x = {{ value = {r.us_per_deg_x:.2f}, {tag}",
        f"us_per_deg_y = {{ value = {r.us_per_deg_y:.2f}, {tag}",
        "[vision]",
        f"plate_yaw_deg = {{ value = {r.plate_yaw_deg:.1f}, {tag}",
        f"px_per_mm = {{ value = {s_ball:.3f}, {tag}",
        f"plate_centre_px = {{ value = [{centre[0]:.1f}, {centre[1]:.1f}], {tag}",
        "",
        f"mirror: {'yes, us_per_deg_y flipped' if r.mirror else 'no'}",
        f"pulses {cal.amps[0]:.0f} and {cal.amps[1]:.0f} us, fit residual {r.resid:.1f} mm/s^2",
        f"friction intercept x {r.friction_x:.1f} y {r.friction_y:.1f} mm/s^2",
        f"one amplitude k x {r.k_one_x:.3f} y {r.k_one_y:.3f} mm/s^2 per us",
        f"camera height check (coarse) {h}",
    ]
    return "\n".join(lines)


def run(bench, cam, cfg, clock=time.monotonic, sleep=time.sleep, ask=input) -> str:
    period = 1 / cfg["control"]["sim_tick_hz"]
    rig = cfg["rig"]
    try:
        bench.submit("l")
        centre, s_ball = centre_and_scale(cam, cfg, clock, sleep, ask, period)
        cal = Autocal(cfg, centre, s_ball)
        pulses(bench, cam, cal, clock, sleep, period)
        return report(cal, centre, s_ball, cfg)
    finally:
        cx, cy = (round(v) for v in center(rig))
        if bench.last_p not in (None, (cx, cy)):
            bench.submit(f"P {cx} {cy}")
        kill_and_wait(bench, period)
        cam.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="fake rig, no board or camera")
    args = ap.parse_args()
    cfg = load()
    if args.sim:
        from src.sim.rig import SimLink, SimRig

        world = SimRig(cfg)
        bench, cam = Bench(SimLink(world), cfg["rig"]), world
        timing = (world.clock, world.advance, lambda _: "")
    else:
        from src.io.camera import CameraSource, open_camera
        from src.io.serial_link import open_board
        from src.vision.pipeline import Pipeline

        bench = Bench(open_board(cfg["rig"]), cfg["rig"])
        cam = CameraSource(open_camera(cfg["rig"]), process=Pipeline(cfg).step)
        timing = (time.monotonic, time.sleep, input)
    try:
        print(run(bench, cam, cfg, *timing))
    except (Abort, KeyboardInterrupt) as e:
        sys.exit(f"stopped: {e or 'ctrl-c'}")


if __name__ == "__main__":
    main()
