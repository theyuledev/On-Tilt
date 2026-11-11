import copy
import math
from pathlib import Path

import numpy as np
import pytest

from src.control import autocal as ac
from src.control.autocal_fit import wrap
from src.control.autocal_hold import DIVERGED
from src.io.settings import BASE, load
from src.sim.rig import TRUTH, SimRig

CFG = load(BASE, Path("no-local-settings.toml"))


def run(
    trims=(0, 0),
    yaw=0.0,
    mirror=False,
    d_mm=61.8,
    quick=False,
    stored=(28.75, 28.75, 0.0),
    hidden=(0.0, 0.0),
    control=None,
    **truth,
):
    cfg = copy.deepcopy(CFG)
    cfg["rig"].update(trim_x_us=0, trim_y_us=0, ball_d_mm=d_mm)
    cfg["rig"].update(us_per_deg_x=stored[0], us_per_deg_y=stored[1])
    cfg["vision"]["plate_yaw_deg"] = stored[2]
    cfg["control"].update(control or {})
    tr = {**TRUTH, "yaw_deg": yaw, "mirror": mirror, "trim_x_us": trims[0], "trim_y_us": trims[1]}
    tr.update(us_per_deg_x=100.0, us_per_deg_y=100.0, friction_mm_s2=47.0, servo_dead_time_s=0.065)
    tr.update(truth)
    world = SimRig(cfg, tr, seed=0)
    cal = ac.Autocal(cfg, tr["centre_px"], tr["s_ball"], quick=quick)
    world.send(*(round(v) for v in cal.c0))
    sent = []
    while world.t < 60:
        world.visible = not hidden[0] <= world.t < hidden[1]
        cmd = cal.step(world.t, world.latest())
        if cmd is None:
            break
        sent.append((cmd.x_us, cmd.y_us))
        world.send(cmd.x_us, cmd.y_us)
        world.advance(1 / cfg["control"]["sim_tick_hz"])
    return cal, np.array(sent)


def max_r(cal) -> float:
    return max(math.hypot(u, v) for _, u, v in cal.rows)


def assert_found(cal, trims, yaw, mirror, upd=100.0):
    r = cal.result
    assert r is not None, cal.aborted
    assert (r.trim_x_us, r.trim_y_us) == pytest.approx(trims, abs=12)
    assert abs(wrap(r.plate_yaw_deg - yaw)) <= 5
    assert r.mirror is mirror
    assert abs(r.us_per_deg_x) == pytest.approx(upd, rel=0.1)
    assert abs(r.us_per_deg_y) == pytest.approx(upd, rel=0.1)


TRIMS = [(0, 0), (15, -25), (60, -60), (-60, 60), (60, 60), (-60, -60)]
YAWS, BALLS = (0, 90, 180, 37), (61.8, 38.0, 13.0)
CASES = [(TRIMS[k % 6], YAWS[k % 4], k % 2 == 1, BALLS[k % 3]) for k in range(12)]


@pytest.mark.parametrize("trims,yaw,mirror,d_mm", CASES)
def test_finds_trims_and_mapping_from_a_wrong_start(trims, yaw, mirror, d_mm):
    cal, sent = run(trims, yaw, mirror, d_mm)
    assert_found(cal, trims, yaw, mirror)
    assert np.abs(sent - cal.c0).max() <= CFG["rig"]["startup_range_us"]
    assert np.abs(cal.level - cal.c0).max() <= CFG["control"]["autocal_level_max_us"]


@pytest.mark.xfail(
    strict=True,
    reason="a 0.10 s chain vs the stored 0.046 s model reads the gain ~24 % off (trims still fine)",
)
@pytest.mark.parametrize("trims,mirror", [((60, -60), False), ((-60, -60), True)])
def test_a_slow_chain_still_finishes(trims, mirror):
    cal, _ = run(trims, 37, mirror, servo_dead_time_s=0.10)
    assert_found(cal, trims, 37, mirror)


def bad_probe(monkeypatch, f):
    fit = ac.probe_fit
    monkeypatch.setattr(ac, "probe_fit", lambda *a: (lambda m, b, r: (f(m), b, r))(*fit(*a)))


TURN = np.array([[0.0, -1.0], [1.0, 0.0]])
FLIPS = [lambda m: -m, lambda m: m[:, ::-1], lambda m: TURN @ m]


@pytest.mark.parametrize("f", FLIPS, ids=["negated", "swapped", "rotated"])
@pytest.mark.parametrize("trims", [(15, -25), (60, -60)])
def test_a_wrong_mapping_stops_inside_the_clamp(monkeypatch, f, trims):
    bad_probe(monkeypatch, f)
    cal, sent = run(trims, 37)
    assert cal.aborted
    assert np.abs(sent - cal.c0).max() <= CFG["rig"]["startup_range_us"]


@pytest.mark.parametrize(
    "f,trims",
    [(FLIPS[0], (15, -25)), (FLIPS[0], (60, -60)), (FLIPS[1], (60, -60)), (FLIPS[2], (60, -60))],
    ids=["negated", "negated-60", "swapped-60", "rotated-60"],
)
def test_a_wrong_mapping_reads_as_diverging(monkeypatch, f, trims):
    bad_probe(monkeypatch, f)
    cal, _ = run(trims, 37)
    assert cal.aborted == DIVERGED


@pytest.mark.parametrize("upd", [100.0, 57.0])
@pytest.mark.parametrize("fric", [47.0, 15.0])
@pytest.mark.parametrize("trims,yaw,mirror", [((60, -60), 37, True), ((-60, 60), 90, False)])
def test_the_right_mapping_never_reads_as_diverging(upd, fric, trims, yaw, mirror):
    cal, _ = run(trims, yaw, mirror, us_per_deg_x=upd, us_per_deg_y=upd, friction_mm_s2=fric)
    assert cal.aborted is None or cal.aborted != DIVERGED


def test_a_stuck_ball_stops_before_the_rim():
    cal, _ = run((60, -60), friction_mm_s2=150.0)
    assert cal.aborted and "rim" not in cal.aborted
    assert max_r(cal) < cal.wall


def test_the_probe_finds_a_slow_chain():
    cal, _ = run((15, -25), 37, servo_dead_time_s=0.10)
    assert cal.dead == pytest.approx(0.10, abs=0.03)


def test_a_ball_that_never_settles_times_out():
    cal, _ = run(control={"hold_v_mm_s": 0.0})
    assert cal.aborted == "could not hold the ball"


@pytest.mark.parametrize("hidden", [(0.0, 0.95), (0.3, 0.7)], ids=["start", "mid-probe"])
def test_dropped_frames_are_a_lost_ball(hidden):
    cal, _ = run(hidden=hidden)
    assert cal.aborted == "lost the ball"


def test_quick_pass_keeps_the_stored_mapping_and_finds_trims():
    cal, _ = run((40, -30), 37, True, quick=True, stored=(100.0, -100.0, 37.0))
    assert cal.note == ""
    assert len(cal.m) == 4
    r = cal.result
    assert (r.trim_x_us, r.trim_y_us) == pytest.approx((40, -30), abs=12)
    assert (r.us_per_deg_x, r.us_per_deg_y, r.plate_yaw_deg, r.mirror) == (100, -100, 37, True)


def test_quick_pass_with_a_wrong_stored_yaw_recalibrates():
    cal, _ = run((40, -30), 37, quick=True, stored=(100.0, 100.0, 217.0))
    assert cal.note == "recalibrating"
    assert len(cal.m) == 8
    assert_found(cal, (40, -30), 37, False)
