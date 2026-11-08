import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import BallState, ServoCommand
from src.io.settings import load
from src.sim.ball import roll
from src.sim.fit import pulse_fit
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

LOG = Path(__file__).resolve().parents[2] / "data" / "runs" / "20261108-025857.jsonl"
WINDOW_S = 0.6
LIMIT_MM = 80
FIT_S = 0.4
FIT_LIMIT_MM = 95
DT = 0.01
DELAYS_S = np.arange(0.0, 0.17, 0.02)


def load_log(path):
    rows = [json.loads(line) for line in path.open()]
    t = np.array([r["t_s"] for r in rows])
    pulses = np.array([(r["servo_cmd"]["x_us"], r["servo_cmd"]["y_us"]) for r in rows])
    armed = np.array([r["armed"] for r in rows])
    seen = {}
    for r in rows:
        b = r["ball"]
        if r["armed"] and b["valid"]:
            seen[b["t_s"]] = (b["x_mm"], b["y_mm"])
    bt = np.array(sorted(seen))
    return t, pulses, armed, bt, np.array([seen[k] for k in bt])


def runs(bt):
    gaps = np.flatnonzero(np.diff(bt) > 0.035)
    return zip(np.r_[0, gaps + 1], np.r_[gaps, bt.size - 1], strict=True)


def windows(bt, xy, step_s=1.0):
    out = []
    for s, e in runs(bt):
        t0 = bt[s]
        while t0 + WINDOW_S <= bt[e]:
            k = (bt >= t0) & (bt <= t0 + WINDOW_S)
            if np.all(np.abs(xy[k]) < LIMIT_MM):
                out.append(np.flatnonzero(k))
            t0 += step_s
    return out


def fit_windows(log, delay):
    t, pulses, _, bt, xy = log
    n, out = round(FIT_S / DT), []
    for s, e in runs(bt):
        g = np.arange(bt[s], bt[e], DT)
        pos = np.c_[
            np.interp(g, bt[s : e + 1], xy[s : e + 1, 0]),
            np.interp(g, bt[s : e + 1], xy[s : e + 1, 1]),
        ]
        p = pulses[np.clip(np.searchsorted(t, g - delay, side="right") - 1, 0, len(t) - 1)]
        for i in range(0, g.size - n, n // 2):
            w = slice(i, i + n)
            if np.all(np.abs(pos[w]) < FIT_LIMIT_MM):
                out.append((pos[w, 0], pos[w, 1], p[w, 0], p[w, 1]))
    return out


def live_level(log):
    # what went out while disarmed is the level pulse with the trims that were live
    _, pulses, armed, _, _ = log
    level = []
    for i in (0, 1):
        values, counts = np.unique(pulses[~armed, i], return_counts=True)
        level.append(float(values[np.argmax(counts)]))
    return level


def configs(log, fitted):
    cfg = load()
    rig, control = dict(cfg["rig"]), dict(cfg["control"])
    level = live_level(log)
    rig["trim_x_us"] = level[0] - rig["center_x_us"]
    rig["trim_y_us"] = level[1] - rig["center_y_us"]
    if not fitted:
        return rig, control, np.eye(2)
    g = control["sim_g_mm_s2"]
    fits = {d: pulse_fit(fit_windows(log, d), DT, level[0], g) for d in DELAYS_S}
    delay, fit = max(fits.items(), key=lambda kv: kv[1].r2)
    rig["trim_x_us"] = fit.level_us[0] - rig["center_x_us"]
    rig["trim_y_us"] = fit.level_us[1] - rig["center_y_us"]
    rig["us_per_deg_x"], rig["us_per_deg_y"] = fit.us_per_deg
    # the fitted delay is a pure one, the servo lag already holds tau of it
    control["servo_dead_time_s"] = max(delay - control["servo_tau_s"], 0.0)
    rad = np.radians(fit.axis_deg)
    return rig, control, np.array([np.cos(rad), np.sin(rad)])


def replay(log, idx, rig, control, axes, warm_s=0.5):
    t, pulses, _, bt, xy = log
    tt, pos = bt[idx], xy[idx]
    start = np.polyfit(tt[:10] - tt[0], pos[:10], 2)
    to_plate = np.linalg.inv(axes)
    p0, v0 = to_plate @ start[2], to_plate @ start[1]
    act = SimActuator(rig, control)
    ball = BallState(p0[0], p0[1], v0[0], v0[1], valid=True, t_s=tt[0])
    path = []
    for k in np.flatnonzero((t >= tt[0] - warm_s) & (t <= tt[-1])):
        state = act.send(ServoCommand(*map(int, pulses[k])), t[k])
        if t[k] > tt[0]:
            ball = roll(ball, state, t[k], rig, control)
            path.append((t[k], *(axes @ (ball.x_mm, ball.y_mm))))
    path = np.array(path)
    sim = np.c_[np.interp(tt, path[:, 0], path[:, 1]), np.interp(tt, path[:, 0], path[:, 2])]
    keep = tt > path[0, 0]
    return float(np.sqrt(np.mean(np.sum((sim[keep] - pos[keep]) ** 2, axis=1))))


def errors(log, fitted):
    rig, control, axes = configs(log, fitted)
    return [replay(log, idx, rig, control, axes) for idx in windows(log[3], log[4])]


@pytest.mark.skipif(not LOG.exists(), reason="run logs not on this machine")
def test_fitted_sim_follows_the_logged_ball_far_closer_than_the_settings_sim():
    log = load_log(LOG)
    now, fitted = np.mean(errors(log, False)), np.mean(errors(log, True))
    assert fitted < 0.5 * now
    assert fitted < 7
