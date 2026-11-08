import math
import random

import pytest

from src.contracts import BallState, Intent
from src.control import ROLL
from src.control.pulse_map import center, pulse_to_tilt
from src.io.settings import load
from src.loop import Loop
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

DT = 0.01
CATCHES = [(60, 0), (0, 60), (60, 60), (-60, 60), (40, -80), (100, 0), (-90, -40), (0, -40)]


class Hand:
    seen = calibrating = False
    progress = 0.0

    def __init__(self, x=0.0):
        self.x, self.t = x, 0.0

    def latest(self):
        return Intent(self.x, 0.0, True, self.t)


class Plant:
    """The real plate behind the Loop: servos on the true trims and dead time, the ball rolling
    in the true frame and seen through a camera mapping that may be yawed or mirrored."""

    def __init__(self, cfg, start, yaw=0.0, mirror=False, trim=(0, 0), dead=None, fric=0.0,
                 noise=0.15, gap=None, v0=(0.0, 0.0), seed=0):
        self.cfg = cfg
        rig = dict(cfg["rig"], trim_x_us=cfg["rig"]["trim_x_us"] + trim[0])
        rig["trim_y_us"] += trim[1]
        control = dict(cfg["control"])
        if dead is not None:
            control["servo_dead_time_s"] = dead
        self.servo = SimActuator(rig, control)
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        f = -1 if mirror else 1
        self.m = ((c, -s * f), (s, c * f))
        self.pos, self.vel = list(start), list(v0)
        self.fric, self.noise, self.gap = fric, noise, gap
        self.rnd = random.Random(seed)
        self.seen = []
        self.pulses = []
        self.wall = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2

    def look(self, now):
        x, y = (p + self.rnd.gauss(0, self.noise) for p in self.pos)
        self.seen.append((now, x, y))
        old = self.seen[max(0, len(self.seen) - 5)]
        span = max(now - old[0], DT)
        valid = not (self.gap and self.gap[0] <= now < self.gap[1])
        return BallState(x, y, (x - old[1]) / span, (y - old[2]) / span, valid, now)

    def send(self, cmd, now):
        state = self.servo.send(cmd, now)
        self.pulses.append(cmd)
        if cmd is not None:
            # the app works the sent tilt out from its own trims, like the serial link does
            sent = pulse_to_tilt(cmd, self.cfg["rig"])
            state = type(state)(sent.tilt_x_deg, sent.tilt_y_deg, state.est_x_deg, state.est_y_deg)
        g = ROLL * self.cfg["control"]["sim_g_mm_s2"]
        a = [g * math.sin(math.radians(d)) for d in (state.est_x_deg, state.est_y_deg)]
        speed = math.hypot(*self.vel)
        for k in range(2):
            acc = self.m[k][0] * a[0] + self.m[k][1] * a[1]
            if speed > 1:
                acc -= self.fric * self.vel[k] / speed
            self.vel[k] += acc * DT
            self.pos[k] += self.vel[k] * DT
            if abs(self.pos[k]) > self.wall:
                self.pos[k], self.vel[k] = math.copysign(self.wall, self.pos[k]), 0.0
        return state


def run(start, mode="balance", seconds=3.0, intent_x=0.0, **kw):
    """Arms the Loop on the plant at t 0; returns (first trip time or None, loop, plant)."""
    cfg = load()
    plant = Plant(cfg, start, **kw)
    hand = Hand(intent_x)
    loop = Loop(cfg, hand, actuator=plant)
    loop.mode = mode
    loop.arm()
    trip = None
    for i in range(round(seconds / DT)):
        now = hand.t = i * DT
        loop.ball = plant.look(now)
        loop.tick(now, DT)
        if trip is None and loop.mapping.bad:
            trip = now
    return trip, loop, plant


RIGHT = {
    "plain": {},
    "trim +45/-45 us": {"trim": (45, -45)},
    "friction 40 mm/s2": {"fric": 40},
    "dead time 0.08 s": {"dead": 0.08},
    "thrown at the wall": {"v0": (600, 0)},
    "lost ball 0.12 s": {"gap": (0.5, 0.62)},
    "noise 0.4 mm": {"noise": 0.4},
}


@pytest.mark.parametrize("case", RIGHT)
def test_the_right_mapping_never_trips(case):
    for i, start in enumerate(CATCHES):
        trip, loop, _ = run(start, seed=i, **RIGHT[case])
        assert trip is None, (case, start, loop.mapping.bad)


WRONG = {"yaw 180": {"yaw": 180}, "yaw 90": {"yaw": 90}, "mirror y": {"mirror": True}}


@pytest.mark.parametrize("case", WRONG)
def test_a_wrong_mapping_trips_fast(case):
    for i, start in enumerate(CATCHES):
        axes = "y" if "mirror" in case else "xy"
        if max(abs(start[k]) for k in range(2) if "xy"[k] in axes) < 40:
            continue
        trip, loop, _ = run(start, seed=i, **WRONG[case])
        assert trip is not None and trip <= 0.6, (case, start, trip)


def test_yaw_30_still_pushes_the_right_way():
    for i, start in enumerate(CATCHES):
        assert run(start, seed=i, yaw=30)[0] is None, start


def test_hand_mode_only_warns():
    trip, loop, plant = run((0, 0), mode="hand", intent_x=1.0, yaw=180, seconds=1.5)
    assert trip is not None and loop.snapshot()["mapping_bad"]
    cx = center(loop.cfg["rig"])[0]
    assert plant.pulses[-1].x_us > cx + 20


def test_a_trip_levels_through_the_slew_and_nothing_kills():
    trip, loop, plant = run((100, 0), yaw=180, seconds=2.0)
    rig = loop.cfg["rig"]
    assert trip is not None and not loop.safety.killing and loop.safety.armed
    after = [p for p in plant.pulses[round(trip / DT):] if p is not None]
    steps = [abs(b.x_us - a.x_us) for a, b in zip(after[:-1], after[1:], strict=True)]
    assert max(steps) <= rig["slew_us_s"] * DT + 1
    assert (after[-1].x_us, after[-1].y_us) == tuple(round(c) for c in center(rig))


def test_rearming_clears_the_flag_and_starts_balance_fresh():
    _, loop, _ = run((100, 0), yaw=180, seconds=1.0)
    assert loop.mapping.bad
    old = loop.step
    loop.arm()
    assert loop.mapping.bad is None and loop.snapshot()["mapping_bad"] is None
    assert loop.step is not old
