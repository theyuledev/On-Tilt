"""Intents from a real hand run on the rig, cut to one armed stretch."""

import csv
from itertools import pairwise
from typing import NamedTuple

import pytest

from src.contracts import Intent, RunLogRecord
from src.control.pulse_map import center
from src.io.settings import ROOT, load
from src.loop import Loop
from src.safety import limits

pytestmark = pytest.mark.replay

RUN = ROOT / "tests/fixtures/hand_run.csv"
T0 = 1000.0  # monotonic clock at the first row
SLACK_S = 0.05  # a few ticks on top of the ramp time


class Replayed:
    seen = True
    calibrating = False
    progress = 1.0

    def __init__(self):
        self.intent = Intent(x=0.0, y=0.0, valid=False, t_s=0.0)

    def latest(self) -> Intent:
        return self.intent

    def recentre(self) -> None:
        pass


class Tick(NamedTuple):
    dt: float
    rec: RunLogRecord
    stale: bool


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[dict, list[Tick]]:
    # checked in numbers only, the conftest one is per test
    cfg = load(local=tmp_path_factory.mktemp("cfg") / "settings.local.toml")
    cfg["control"]["mode"] = "hand"
    hand = Replayed()
    loop = Loop(cfg, hand=hand)
    loop.arm()
    ticks: list[Tick] = []

    def tick(now: float) -> None:
        dt = now - ticks[-1].rec.t_s if ticks else 0.0
        ticks.append(Tick(dt, loop.tick(now, dt), loop.stale))

    with RUN.open() as f:
        for row in csv.DictReader(f):
            now = T0 + float(row["t"])
            hand.intent = Intent(float(row["x"]), float(row["y"]), row["valid"] == "1", now)
            tick(now)
    # then the hand goes quiet, for long enough that the plate has to be home
    rig = cfg["rig"]
    walk_s = max(rig["limit_x_us"], rig["limit_y_us"]) / rig["kill_ramp_us_s"]
    end = now + rig["stale_intent_s"] + walk_s + 2 * SLACK_S
    step = (now - T0) / (len(ticks) - 1)
    while now < end:
        now += step
        tick(now)
    return cfg, ticks


def pulse(tick: Tick) -> tuple[int, int]:
    return tick.rec.servo_cmd.x_us, tick.rec.servo_cmd.y_us


def stale_stretches(ticks: list[Tick]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, t in enumerate([*ticks, None]):
        stale = t is not None and t.stale and t.rec.armed
        if stale and start is None:
            start = i
        elif not stale and start is not None:
            out.append((start, i))
            start = None
    return out


def test_every_sent_pulse_stays_inside_the_limits(run):
    cfg, ticks = run
    x_lo, x_hi, y_lo, y_hi = limits(cfg["rig"])
    assert all(t.rec.armed for t in ticks)
    for t in ticks:
        x, y = pulse(t)
        assert x_lo <= x <= x_hi and y_lo <= y <= y_hi, t.rec.t_s


def test_sent_pulse_never_steps_faster_than_the_ramps(run):
    cfg, ticks = run
    rig = cfg["rig"]
    rate = max(rig["slew_us_s"], rig["kill_ramp_us_s"])
    for a, b in pairwise(ticks):
        for pa, pb in zip(pulse(a), pulse(b), strict=True):
            assert abs(pb - pa) <= rate * b.dt + 1, b.rec.t_s


def test_lost_or_stale_hand_walks_the_plate_home(run):
    cfg, ticks = run
    rig = cfg["rig"]
    centre = center(rig)
    stretches = stale_stretches(ticks)
    assert stretches
    for a, b in stretches:
        far = [abs(p - c) for p, c in zip(pulse(ticks[a - 1]), centre, strict=True)]
        due = ticks[a - 1].rec.t_s + (max(far) + 1) / rig["kill_ramp_us_s"] + SLACK_S
        for t in ticks[a:b]:
            d = [abs(p - c) for p, c in zip(pulse(t), centre, strict=True)]
            assert d[0] <= far[0] and d[1] <= far[1], t.rec.t_s
            if t.rec.t_s >= due:
                assert d == [0, 0], t.rec.t_s
            far = d
    assert pulse(ticks[-1]) == centre


def test_plate_follows_the_player_both_ways_on_both_axes(run):
    cfg, ticks = run
    cx, cy = center(cfg["rig"])
    live = [pulse(t) for t in ticks if t.rec.armed and not t.stale and t.rec.intent.valid]
    xs, ys = [p[0] for p in live], [p[1] for p in live]
    assert min(xs) < cx < max(xs)
    assert min(ys) < cy < max(ys)
