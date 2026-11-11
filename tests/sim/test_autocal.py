import copy
import re
import sys
from pathlib import Path

import pytest

from src.control import autocal as stepper
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


def watch(monkeypatch, link, bench, hook=None) -> dict:
    # the stepper's latest pulse and how many pulses were out before it
    mark = {"n": 0, "cmd": None}

    class Spy(autocal.Autocal):
        def step(self, now, obs):
            cmd = super().step(now, obs)
            mark["n"] = sum(1 for line in link.sent if line[0] == "P")
            if cmd is not None:
                mark["cmd"] = (cmd.x_us, cmd.y_us)
            if hook:
                hook(self)
            return cmd

    monkeypatch.setattr(autocal, "Autocal", Spy)
    return mark


def stops_at_level(link: SimLink, level=(1515, 1475), mark=None) -> None:
    # every pulse stays inside the startup range around level, the last ones are level, then D
    span = CFG["rig"]["startup_range_us"]
    sent = [tuple(int(v) for v in line.split()[1:]) for line in link.sent if line[0] == "P"]
    assert max(abs(p[k] - level[k]) for p in sent for k in (0, 1)) <= span
    last = max(i for i, p in enumerate(sent) if p != level)
    assert sent[last + 1 :] and all(p == level for p in sent[last + 1 :])
    assert link.sent[-1] == "D"
    if mark is not None:
        # past resends of the stepper's last pulse, the next one is level itself, not a ramp step
        rest = [p for p in sent[mark["n"] :]]
        while rest and rest[0] == mark["cmd"]:
            rest.pop(0)
        assert (rest[0] if rest else mark["cmd"]) == level


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
    stops_at_level(link)


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


def test_a_wrong_stored_trim_is_held_and_found():
    cfg, world, link, bench = setup(trims=(0, 0))
    out = autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    assert float(value(out, "trim_x_us")) == pytest.approx(15, abs=12)
    assert float(value(out, "trim_y_us")) == pytest.approx(-25, abs=12)
    stops_at_level(link, level=(1500, 1500))


def interrupt_at(world, after_s: float, hit=None):
    def sleep(dt):
        if world.t >= after_s:
            if hit is None:
                raise KeyboardInterrupt
            hit()
        world.advance(dt)

    return sleep


def test_ctrl_c_mid_pulse_levels_then_disarms(monkeypatch):
    cfg, world, link, bench = setup()
    mark = watch(monkeypatch, link, bench)
    with pytest.raises(KeyboardInterrupt):
        autocal.run(bench, world, cfg, world.clock, interrupt_at(world, 1.4), lambda _: "")
    stops_at_level(link, mark=mark)


def aborts(monkeypatch, match, hide=None, control=None):
    cfg, world, link, bench = setup()
    cfg["control"].update(control or {})

    def hook(cal):
        if hide and hide(cal):
            world.visible = False

    mark = watch(monkeypatch, link, bench, hook)
    with pytest.raises(autocal.Abort, match=match):
        autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    stops_at_level(link, mark=mark)


STATES = {
    "probe": lambda cal: cal.M is None and len(cal.rows) > 30,
    "hold": lambda cal: cal.hold,
    "doublet": lambda cal: cal.M is not None and not cal.hold,
}


@pytest.mark.parametrize("state", STATES)
def test_lost_ball_levels_then_disarms(monkeypatch, state):
    aborts(monkeypatch, "lost", hide=STATES[state])


@pytest.mark.parametrize(
    "control,match",
    [
        ({"autocal_probe_rms_mm": 0.0}, "did not answer the probe"),
        ({"hold_v_mm_s": 0.0}, "could not hold"),
        ({"autocal_rim_margin_mm": 110.0}, "rim"),
    ],
    ids=["probe", "hold-timeout", "rim"],
)
def test_each_abort_levels_then_disarms(monkeypatch, control, match):
    aborts(monkeypatch, match, control=control)


def test_a_wrong_mapping_levels_then_disarms(monkeypatch):
    fit = stepper.probe_fit
    monkeypatch.setattr(stepper, "probe_fit", lambda *a: (lambda m, b, r: (-m, b, r))(*fit(*a)))
    aborts(monkeypatch, "mapping looks wrong")


def test_a_refused_pulse_mid_probe_stops_the_run(monkeypatch):
    cfg, world, link, bench = setup()
    mark = watch(monkeypatch, link, bench)
    answer = link.raw

    def refuse_once(line):
        if line[0] == "P" and world.t >= 1.3 and not link.refused:
            link.refused = True
            link.sent.append(line)
            return "ERR ?"
        return answer(line)

    link.refused, link.raw = False, refuse_once
    with pytest.raises(autocal.Abort, match="ERR"):
        autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")
    stops_at_level(link, mark=mark)


def test_a_missed_heartbeat_disarms_the_board():
    cfg, world, link, bench = setup()
    bench.tick = lambda dt: None
    with pytest.raises(autocal.Abort, match="disarmed"):
        autocal.run(bench, world, cfg, world.clock, world.advance, lambda _: "")


def test_sim_entry_runs_end_to_end(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["autocal", "--sim"])
    monkeypatch.setattr(autocal, "load", lambda: copy.deepcopy(CFG))
    autocal.main()
    out = capsys.readouterr().out
    assert float(value(out, "plate_yaw_deg")) == pytest.approx(37, abs=2)
    assert "mirror: no" in out
    assert re.search(r"probe best dead time 0\.\d\d s vs stored 0.046 s", out)
