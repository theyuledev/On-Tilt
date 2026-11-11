import time

import pytest

from src import safety
from src.app import Loop
from src.contracts import BallState
from src.control.pulse_map import center
from src.io.settings import load

pytestmark = pytest.mark.sim


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def ticks(loop, clock, n, key=None, dt=0.01):
    for _ in range(n):
        clock[0] += dt
        if key:
            loop.keys.set(*key)
        loop.tick(clock[0], dt)


def hold_ball_at(loop, clock, x, n, key=(0.0, 0.0), dt=0.01):
    for _ in range(n):
        loop.ball = BallState(x, 0.0, 0.0, 0.0, valid=True, t_s=clock[0] + dt)
        ticks(loop, clock, 1, key=key, dt=dt)


def test_hand_step_back_to_zero_lands_exactly_on_centre(clock):
    cfg = load()
    cfg["control"]["inverter_on"] = True  # must stay out of the way in hand mode
    loop = Loop(cfg)
    loop.arm()
    ticks(loop, clock, 100, key=(0.9, 0.0))
    assert loop.last.plate_state.est_x_deg > 3
    ticks(loop, clock, 200, key=(0.0, 0.0))
    rig, control = loop.cfg["rig"], loop.cfg["control"]
    assert (loop.last.servo_cmd.x_us, loop.last.servo_cmd.y_us) == center(rig)
    off_us = abs(loop.last.plate_state.est_x_deg) * rig["us_per_deg_x"]
    assert off_us <= control["servo_dead_band_us"] / 2


def test_integral_stays_clear_while_the_intent_is_stale(clock):
    cfg = load()
    cfg["control"]["mode"] = "balance"
    loop = Loop(cfg)
    loop.arm()
    loop.ball = BallState(20.0, 0.0, 0.0, 0.0, valid=True, t_s=clock[0])
    ticks(loop, clock, 50)
    assert loop.stale and loop.step.acc.x == 0
    ticks(loop, clock, 50, key=(0.0, 0.0))
    assert not loop.stale and loop.step.acc.x != 0


def test_a_kill_that_ends_inside_one_tick_still_leaves_on_the_level_pulse(clock):
    cfg = load()
    cfg["control"]["mode"] = "balance"
    cfg["control"]["inverter_on"] = True
    loop = Loop(cfg)
    loop.arm()
    hold_ball_at(loop, clock, 60.0, 30)
    hold_ball_at(loop, clock, 6.0, 30)
    assert loop.band[0].committed != center(cfg["rig"])[0]
    loop.kill()
    hold_ball_at(loop, clock, 6.0, 1, dt=0.05)
    # the 40 ms level hold counts from the first level pulse, not the slow tick that got there
    assert loop.safety.armed
    hold_ball_at(loop, clock, 6.0, 4)
    assert not loop.safety.armed
    assert (loop.last.servo_cmd.x_us, loop.last.servo_cmd.y_us) == center(cfg["rig"])


def test_a_stale_intent_ramps_on_exactly_what_safety_put_out(clock, monkeypatch):
    cfg = load()
    cfg["control"]["mode"] = "balance"
    cfg["control"]["inverter_on"] = True
    loop = Loop(cfg)
    loop.arm()
    hold_ball_at(loop, clock, 60.0, 30)
    hold_ball_at(loop, clock, 6.0, 30)
    assert loop.band[0].committed != center(cfg["rig"])[0] and loop.band[0].direction != 0

    approved = []
    real = safety.step

    def spy(*args):
        approved.append(real(*args))
        return approved[-1]

    monkeypatch.setattr(safety, "step", spy)
    clock[0] += cfg["rig"]["stale_intent_s"]  # no key since, so the intent is stale from here
    for _ in range(10):
        hold_ball_at(loop, clock, 6.0, 1, key=None)
        assert loop.stale and loop.last.servo_cmd == approved[-1]
