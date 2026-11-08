import copy
import re
import sys
from pathlib import Path

import pytest

from src.io.settings import BASE, load
from src.sim.rig import TRUTH, SimLink, SimRig
from tools import autocal
from tools.bench import Bench

CFG = load(BASE, Path("no-local-settings.toml"))


def setup(d_mm=61.8, trims=(15, -25), **truth):
    cfg = copy.deepcopy(CFG)
    cfg["rig"].update(trim_x_us=trims[0], trim_y_us=trims[1], ball_d_mm=d_mm)
    cfg["vision"].update(plate_yaw_deg=0.0, plate_centre_px=[640, 400])
    world = SimRig(cfg, {**TRUTH, "mirror": True, **truth}, seed=7)
    link = SimLink(world)
    return cfg, world, link, Bench(link, cfg["rig"])


def value(out: str, key: str) -> str:
    return re.search(rf"^{key} = {{ value = (\S+?),", out, re.M).group(1)


def ends_level_then_disarms(link: SimLink, start: int, level=(1515, 1475)) -> None:
    tail = link.sent[start:]
    assert tail[-1] == "D"
    assert all(line == "P {} {}".format(*level) for line in tail[:-1] if line.startswith("P"))


@pytest.mark.parametrize("d_mm", [61.8, 38.0, 13.0])
def test_recovers_yaw_mirror_scale_and_gain(d_mm):
    cfg, world, link, bench = setup(d_mm)
    out = autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    assert float(value(out, "trim_x_us")) == pytest.approx(15, abs=12)
    assert float(value(out, "trim_y_us")) == pytest.approx(-25, abs=12)
    assert float(value(out, "plate_yaw_deg")) == pytest.approx(37, abs=2)
    assert float(value(out, "us_per_deg_x")) == pytest.approx(57, rel=0.1)
    assert float(value(out, "us_per_deg_y")) == pytest.approx(-57, rel=0.1)
    assert float(value(out, "px_per_mm")) == pytest.approx(2.6, rel=0.01)
    cx, cy = re.search(r"plate_centre_px = { value = \[(\S+), (\S+)\]", out).groups()
    assert (float(cx), float(cy)) == pytest.approx((700, 380), abs=1)
    assert "mirror: yes" in out
    assert link.sent.index("A") < min(i for i, s in enumerate(link.sent) if s.startswith("P"))
    ends_level_then_disarms(link, len(link.sent) - 5)


def test_second_amplitude_shrinks_on_a_soft_plate():
    cfg, world, link, bench = setup(us_per_deg_x=28.75, us_per_deg_y=28.75)
    out = autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    amps = re.search(r"pulses (\d+) and (\d+) us", out).groups()
    assert int(amps[1]) < cfg["control"]["autocal_a2_max_us"]
    assert float(value(out, "us_per_deg_x")) == pytest.approx(28.75, rel=0.1)


def test_first_pulse_clears_heavy_rolling_friction():
    slow = dict(us_per_deg_x=100.0, us_per_deg_y=100.0, servo_dead_time_s=0.06)
    cfg, world, link, bench = setup(friction_mm_s2=47.0, **slow)
    out = autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    assert float(value(out, "trim_x_us")) == pytest.approx(15, abs=12)
    assert float(value(out, "trim_y_us")) == pytest.approx(-25, abs=12)
    assert float(value(out, "us_per_deg_x")) == pytest.approx(100, rel=0.1)


def test_a_wrong_stored_trim_rolls_the_ball_off_and_stops_safely():
    cfg, world, link, bench = setup(trims=(0, 0))
    with pytest.raises(autocal.Abort, match="rim"):
        autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    tilted = [i for i, s in enumerate(link.sent) if s.startswith("P") and s != "P 1500 1500"]
    ends_level_then_disarms(link, tilted[-1] + 1, level=(1500, 1500))


def interrupt_at(world, after_s: float, hit=None):
    def sleep(dt):
        if world.t >= after_s:
            if hit is None:
                raise KeyboardInterrupt
            hit()
        world.advance(dt)

    return sleep


def test_ctrl_c_mid_pulse_levels_then_disarms():
    cfg, world, link, bench = setup()
    with pytest.raises(KeyboardInterrupt):
        autocal.run(bench, world, cfg, world.clock, interrupt_at(world, 1.4), lambda _: "")
    tilted = [i for i, s in enumerate(link.sent) if s.startswith("P") and s != "P 1515 1475"]
    ends_level_then_disarms(link, tilted[-1] + 1)


def test_lost_ball_levels_then_disarms():
    cfg, world, link, bench = setup()

    def hide():
        world.visible = False

    with pytest.raises(autocal.Abort, match="lost"):
        autocal.run(bench, world, cfg, world.clock, interrupt_at(world, 1.4, hide), lambda _: "")
    tilted = [i for i, s in enumerate(link.sent) if s.startswith("P") and s != "P 1515 1475"]
    ends_level_then_disarms(link, tilted[-1] + 1)


def test_sim_entry_runs_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autocal", "--sim"])
    monkeypatch.setattr(autocal, "load", lambda: copy.deepcopy(CFG))
    autocal.main()
    out = capsys.readouterr().out
    assert float(value(out, "plate_yaw_deg")) == pytest.approx(37, abs=2)
    assert "mirror: no" in out
