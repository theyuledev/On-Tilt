import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import BallState, ServoCommand
from src.io.settings import load
from src.sim.ball import roll
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

LOG = Path(__file__).resolve().parents[2] / "data" / "runs" / "20261108-025857.jsonl"
# fitted from this log (pulse -> ball accel regression): the real level pulse, the accel a pulse
# offset gives, the direction a + pulse rolls the ball in the frame vision had then, total delay
FIT = {
    "level": (1412, 1416),
    "us_per_deg": (97.4, 103.1),
    "axis_deg": (15.5, 113.3),
    "dead_s": 0.06,
}
LIVE_TRIM_US = -80
WINDOW_S = 0.6
LIMIT_MM = 80


def load_log(path):
    rows = [json.loads(line) for line in path.open()]
    t = np.array([r["t_s"] for r in rows])
    pulses = np.array([(r["servo_cmd"]["x_us"], r["servo_cmd"]["y_us"]) for r in rows])
    seen = {}
    for r in rows:
        b = r["ball"]
        if r["armed"] and b["valid"]:
            seen[b["t_s"]] = (b["x_mm"], b["y_mm"])
    bt = np.array(sorted(seen))
    return t, pulses, bt, np.array([seen[k] for k in bt])


def windows(bt, xy, step_s=1.0):
    gaps = np.flatnonzero(np.diff(bt) > 0.035)
    out = []
    for s, e in zip(np.r_[0, gaps + 1], np.r_[gaps, bt.size - 1], strict=True):
        t0 = bt[s]
        while t0 + WINDOW_S <= bt[e]:
            k = (bt >= t0) & (bt <= t0 + WINDOW_S)
            if np.all(np.abs(xy[k]) < LIMIT_MM):
                out.append(np.flatnonzero(k))
            t0 += step_s
    return out


def configs(fitted):
    cfg = load()
    rig, control = dict(cfg["rig"]), dict(cfg["control"])
    rig["trim_x_us"] = rig["trim_y_us"] = LIVE_TRIM_US
    axes = np.eye(2)
    if fitted:
        rig["trim_x_us"] = FIT["level"][0] - rig["center_x_us"]
        rig["trim_y_us"] = FIT["level"][1] - rig["center_y_us"]
        rig["us_per_deg_x"], rig["us_per_deg_y"] = FIT["us_per_deg"]
        control["servo_dead_time_s"] = FIT["dead_s"]
        rad = np.radians(FIT["axis_deg"])
        axes = np.array([np.cos(rad), np.sin(rad)])
    return rig, control, axes


def replay(log, idx, rig, control, axes, warm_s=0.5):
    t, pulses, bt, xy = log
    tt, pos = bt[idx], xy[idx]
    fit = np.polyfit(tt[:10] - tt[0], pos[:10], 2)
    to_plate = np.linalg.inv(axes)
    p0, v0 = to_plate @ fit[2], to_plate @ fit[1]
    act = SimActuator(rig, control)
    ball = BallState(p0[0], p0[1], v0[0], v0[1], valid=True, t_s=tt[0])
    ks = np.flatnonzero((t >= tt[0] - warm_s) & (t <= tt[-1]))
    path = []
    for k in ks:
        state = act.send(ServoCommand(*map(int, pulses[k])), t[k])
        if t[k] > tt[0]:
            ball = roll(ball, state, t[k], rig, control)
            path.append((t[k], *(axes @ (ball.x_mm, ball.y_mm))))
    path = np.array(path)
    sim = np.c_[np.interp(tt, path[:, 0], path[:, 1]), np.interp(tt, path[:, 0], path[:, 2])]
    keep = tt > path[0, 0]
    return float(np.sqrt(np.mean(np.sum((sim[keep] - pos[keep]) ** 2, axis=1))))


def errors(log, fitted):
    rig, control, axes = configs(fitted)
    return [replay(log, idx, rig, control, axes) for idx in windows(log[2], log[3])]


@pytest.mark.skipif(not LOG.exists(), reason="run logs not on this machine")
def test_fitted_sim_follows_the_logged_ball_far_closer_than_the_settings_sim():
    log = load_log(LOG)
    now, fitted = np.mean(errors(log, False)), np.mean(errors(log, True))
    # 108 windows: settings 14.7 mm, fitted 6.2 mm
    assert fitted < 0.5 * now
    assert fitted < 8
