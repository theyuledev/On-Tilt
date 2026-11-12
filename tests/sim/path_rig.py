"""Path mode on a sim plate made as awkward as the real one: the ball is seen late, lost for
stretches, and sticks until the plate leans enough."""

import functools
import math
import random
import statistics
from dataclasses import replace

from src import safety
from src.contracts import BallState, Intent
from src.control.pulse_map import tilt_to_pulse
from src.io.settings import load
from src.modes import make_mode
from src.modes.path import project, seg_len
from src.sim.ball import roll
from src.sim.plate import SimActuator

SEEN_LATE_S = 0.12  # command to seen motion, the servo's dead time included
START_DEG = 1.5  # lean a resting ball needs before it rolls
ROUGH_MM_S2 = 47.0  # rolling friction
GAPS_S = ((2.0, 0.4), (5.0, 0.9))  # (from, for) stretches with no ball
NOISE_MM = 0.15
NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)
# the rest zone and smoothing balance runs with on the real plate
RIG_CONTROL = {
    "vel_ema_s": 0.1,
    "balance_rest_mm": 5,
    "balance_rest_out_mm": 24,
    "balance_rest_mm_s": 1000,
    "balance_rest_trim": 0.0,
    "balance_rest_still_mm_s": 4,
    "balance_rest_soft": True,
}


def rig_cfg(path, control=None) -> dict:
    cfg = load()
    # the small ball on the real board; at these pulses per degree the limit is about 2 degrees
    cfg["rig"].update(ball_d_mm=16.0, plate_inner_mm=292.0, us_per_deg_x=72.0, us_per_deg_y=74.0)
    cfg["control"].update(control or {}, mode="path", path_mm=[list(p) for p in path])
    return cfg


def bursts(seconds, seed=2):
    """Gaps of 0.1 to 0.3 s that leave the ball unseen for about a third of the time."""
    rng, t, out = random.Random(seed), 0.0, []
    while t < seconds:
        t += rng.uniform(0.2, 0.6)
        n = rng.uniform(0.1, 0.3)
        out.append((t, n))
        t += n
    return tuple(out)


def sticky_roll(ball, plate, now, rig, control, lean=(0.0, 0.0)):
    plate = replace(plate, est_x_deg=plate.est_x_deg + lean[0], est_y_deg=plate.est_y_deg + lean[1])
    held = math.hypot(plate.est_x_deg, plate.est_y_deg) < START_DEG
    if held and ball.vx_mm_s == 0.0 and ball.vy_mm_s == 0.0:
        return replace(ball, t_s=now)
    b = roll(ball, plate, now, rig, control)
    speed = math.hypot(b.vx_mm_s, b.vy_mm_s)
    drop = ROUGH_MM_S2 * (now - ball.t_s)
    turned = b.vx_mm_s * ball.vx_mm_s + b.vy_mm_s * ball.vy_mm_s < 0
    if speed <= drop or (turned and held):
        return replace(b, vx_mm_s=0.0, vy_mm_s=0.0)
    k = (speed - drop) / speed
    return replace(b, vx_mm_s=b.vx_mm_s * k, vy_mm_s=b.vy_mm_s * k)


def off_line(p, path) -> float:
    best = math.inf
    for a, b in zip(path, path[1:], strict=False):
        f = project(p, a, b) / seg_len(a, b)
        best = min(best, math.dist(p, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))))
    return best


def seen_late(seen, ball, rng, lag, dt, lost) -> BallState:
    """The ball as the mode gets it: lag ticks old, noisy, its speed taken over four samples,
    and not valid while it is lost."""
    seen.append((ball.t_s, ball.x_mm + rng.gauss(0, NOISE_MM), ball.y_mm + rng.gauss(0, NOISE_MM)))
    t, x, y = seen[max(len(seen) - 1 - lag, 0)]
    ta, xa, ya = seen[max(len(seen) - 4 - lag, 0)]
    span = max(t - ta, dt)
    return BallState(x, y, (x - xa) / span, (y - ya) / span, valid=not lost, t_s=t)


def mark(out, off, ball, now, path, end, capture, said_done) -> None:
    """Brings the measures of a run up to date with where the ball really is at this tick."""
    at = (ball.x_mm, ball.y_mm)
    out["far_mm"] = max(out["far_mm"], abs(ball.x_mm), abs(ball.y_mm))
    if said_done and math.isnan(out["done_s"]):
        out["done_s"] = now
    if not out["finished"]:
        off.append(off_line(at, path))
        if math.dist(at, end) < capture:
            out.update(finished=True, time_s=now)
            out["finish_mm_s"] = math.hypot(ball.vx_mm_s, ball.vy_mm_s)


def follow(
    path,
    start=None,
    end=None,
    seconds=60.0,
    rest_s=10.0,
    gaps=GAPS_S,
    lean=(0.0, 0.0),
    cfg=None,
    mode=None,
    track=None,
):
    """Runs a mode (default path mode) from start (default the first point) and measures the
    run from where the ball really is; the finish is the true ball within capture of end.
    track, when given, collects where the ball really was at every tick."""
    cfg = cfg or rig_cfg(path)
    rig, control = cfg["rig"], cfg["control"]
    dt = 1 / control["sim_tick_hz"]
    lag = round((SEEN_LATE_S - control["servo_dead_time_s"]) / dt)
    plate, guard, rng = SimActuator(rig, control), safety.SafetyState(), random.Random(1)
    guard.arm()
    mode = mode or make_mode("path", cfg)
    end = end or path[-1]
    ball = BallState(*(start or path[0]), 0.0, 0.0, valid=True, t_s=0.0)
    seen, off = [], []
    out = {"finished": False, "time_s": math.nan, "finish_mm_s": math.nan, "done_s": math.nan}
    out.update(lost_ticks=0, far_mm=0.0)
    for i in range(1, round(seconds / dt) + 1):
        now = i * dt
        if out["finished"] and now > out["time_s"] + rest_s:
            break
        lost = any(g <= now < g + n for g, n in gaps)
        out["lost_ticks"] += lost
        cmd = mode(seen_late(seen, ball, rng, lag, dt, lost), NO_INTENT, cfg)
        state = plate.send(safety.step(guard, tilt_to_pulse(cmd, rig), False, dt, rig), now)
        ball = sticky_roll(ball, state, now, rig, control, lean)
        if track is not None:
            track.append((ball.x_mm, ball.y_mm))
        said_done = getattr(mode, "done", False)
        mark(out, off, ball, now, path, end, control["path_capture_mm"], said_done)
    out.update(off_max_mm=max(off), off_median_mm=statistics.median(off))
    out.update(end_mm=math.dist((ball.x_mm, ball.y_mm), end))
    out.update(end_mm_s=math.hypot(ball.vx_mm_s, ball.vy_mm_s))
    return out


def line(a, b, step=10.0):
    n = max(round(math.dist(a, b) / step), 1)
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]


def spiral_r(a, r0=30.0, pitch=17.0):
    return r0 + pitch * a / (2 * math.pi)


def spiral(turns=4.0, step=10.0):
    """About 1.6 m outward with 17 mm between its turns: the points and the angle at each."""
    points, angles, a = [], [], 0.0
    while a <= turns * 2 * math.pi:
        r = spiral_r(a)
        points.append((r * math.cos(a), r * math.sin(a)))
        angles.append(a)
        a += step / r
    return points, angles


STRAIGHT = line((-100.0, -40.0), (100.0, 40.0))
S_CURVE = [(float(x), 40.0 * math.sin(math.pi * x / 100)) for x in range(-100, 101, 10)]
CORNER = line((-90.0, 70.0), (40.0, 70.0)) + line((40.0, 70.0), (40.0, -80.0))[1:]
PATHS = {"straight": STRAIGHT, "s curve": S_CURVE, "corner": CORNER}


# case -> ([control] keys on top of the defaults, a third of the time blind instead of two gaps)
CASES = {"default": (None, False), "blind 1/3": (None, True), "rig keys": (RIG_CONTROL, False)}


@functools.cache
def run_case(case, name) -> dict:
    control, blind = CASES[case]
    path = PATHS[name]
    return follow(path, gaps=bursts(60.0) if blind else GAPS_S, cfg=rig_cfg(path, control))


def table():
    rows = ["case       path      off max  median  finished  time s  mm/s there  rests mm  at mm/s"]
    for case in CASES:
        for name in PATHS:
            r = run_case(case, name)
            rows.append(
                f"{case:10s} {name:9s} {r['off_max_mm']:7.1f} {r['off_median_mm']:7.1f}  "
                f"{r['finished']!s:8s} {r['time_s']:7.1f} {r['finish_mm_s']:11.1f} "
                f"{r['end_mm']:9.1f} {r['end_mm_s']:8.1f}"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    print(table())
