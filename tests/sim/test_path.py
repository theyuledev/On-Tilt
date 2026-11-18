import math
import time
from dataclasses import replace

import pytest

from src.app import Loop
from src.control.pid import Integral
from src.games import maze, scoring
from src.games.session import Game
from src.io.settings import load
from src.modes import MODES

pytestmark = pytest.mark.sim


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def follow(cfg, clock, seconds, loop=None):
    loop = loop or Loop(cfg)
    loop.arm()
    dt = 1 / cfg["control"]["sim_tick_hz"]
    for _ in range(round(seconds / dt)):
        clock[0] += dt
        # a live but idle intent, otherwise the stale guard levels the plate
        loop.keys.set(0.0, 0.0)
        yield loop.tick(clock[0], dt).ball


def lapped(cfg, clock, seconds, loop=None):
    """Runs the default square and checks every tick that the ball stays off the rim and the
    path is never done; the corners in the order they were reached, and the last view."""
    rig, loop = cfg["rig"], loop or Loop(cfg)
    edge = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2
    square, hit = [tuple(p) for p in cfg["control"]["path_mm"]], []
    for ball in follow(cfg, clock, seconds, loop):
        assert (
            abs(ball.x_mm) < edge
            and abs(ball.y_mm) < edge
            and not getattr(loop.step, "done", False)
        )
        near = [
            c for c in square[1:] if math.dist((ball.x_mm, ball.y_mm), c) < rig["ball_d_mm"] / 2
        ]
        if near and (not hit or hit[-1] != near[0]):
            hit.append(near[0])
    return hit, loop.snapshot()["path"]


def test_path_mode_laps_the_square_corner_by_corner_until_disarmed(clock):
    cfg = load()
    cfg["control"]["mode"] = "path"
    hit, view = lapped(cfg, clock, 60.0)
    corners = [tuple(p) for p in cfg["control"]["path_mm"][1:]]
    assert hit == (corners * 9)[: len(hit)] and len(hit) >= 12
    assert view["laps"] >= 3 and not view["done"]


def test_attract_mode_laps_the_default_square_and_is_never_done(clock):
    cfg = load()
    cfg["game"]["attract"] = True
    game = Game(cfg, maze.load(cfg["game"]["maze"]), scoring.step)
    loop = Loop(cfg, game=game)
    loop.bus.subscribe("tick", lambda rec: game.tick(rec, loop.safety.killing, loop.board_lost))
    hit, view = lapped(cfg, clock, 30.0, loop)
    assert loop.mode == "path" and game.mode() == "path" and len(hit) >= 4
    assert view["laps"] >= 1 and not view["done"]


def test_a_one_point_path_parks_the_ball_on_the_point(clock):
    cfg = load()
    cfg["control"]["mode"] = "path"
    cfg["control"]["path_mm"] = [[50, -30]]
    *_, ball = follow(cfg, clock, 15.0)
    assert math.dist((ball.x_mm, ball.y_mm), (50, -30)) < cfg["control"]["path_capture_mm"]


def test_mode_is_picked_by_name_and_arming_again_starts_the_path_over():
    loop = Loop(load())
    assert loop.step is MODES["hand"]
    loop.mode = "path"
    loop.step.s, loop.step.laps = 120.0, 2
    loop.arm()
    assert loop.mode == "path" and (loop.step.s, loop.step.laps) == (0.0, 0)


OPEN = [[x, 0] for x in range(0, 101, 10)]


def armed(clock, **control):
    cfg = load()
    cfg["control"].update(control)
    loop = Loop(cfg)
    loop.arm()
    return loop, 1 / cfg["control"]["sim_tick_hz"]


def run(loop, clock, dt, seconds, keys=True):
    for _ in range(round(seconds / dt)):
        clock[0] += dt
        if keys:
            loop.keys.set(0.0, 0.0)
        loop.tick(clock[0], dt)


def from_end(loop):
    return math.dist((loop.ball.x_mm, loop.ball.y_mm), OPEN[-1])


def test_an_open_path_run_goes_on_after_a_gap_in_the_heartbeat(clock):
    loop, dt = armed(clock, mode="path", path_mm=OPEN)
    run(loop, clock, dt, 1.0)
    step, s = loop.step, loop.step.s
    assert s > 0 and not step.done
    run(loop, clock, dt, 0.5, keys=False)
    assert loop.stale and loop.step is step
    run(loop, clock, dt, 10.0)
    assert step.done and from_end(loop) < 15
    # a finished ball stays finished: the gap neither turns the path round nor forgets it is
    # done, also with the ball outside the capture ring where nothing would set done again
    step.hold.acc = Integral(0.5, -0.5, clock[0])
    run(loop, clock, dt, 1.0, keys=False)
    loop.ball = replace(loop.ball, x_mm=OPEN[-1][0] - 18, vx_mm_s=0.0, vy_mm_s=0.0)
    run(loop, clock, dt, 0.5, keys=False)
    assert loop.step is step and step.done
    assert (step.hold.acc.x, step.hold.acc.y) == (0.0, 0.0)
    run(loop, clock, dt, 5.0)
    view = loop.snapshot()["path"]
    assert view["done"] and view["points"] == OPEN and view["chase"] == OPEN[-1]
    assert view["along_mm"] == round(step.s) > 0
    assert from_end(loop) < 15


def test_a_killed_open_path_run_starts_fresh(clock):
    loop, dt = armed(clock, mode="path", path_mm=OPEN)
    run(loop, clock, dt, 1.0)
    step = loop.step
    loop.kill()
    run(loop, clock, dt, 0.05)
    assert loop.step is not step and loop.step.s == 0.0
    # and with no heartbeat either: only an armed run is kept
    run(loop, clock, dt, 0.5, keys=False)
    step = loop.step
    run(loop, clock, dt, 0.1, keys=False)
    assert loop.stale and loop.step is not step


def test_a_lap_goes_on_where_it_was_after_a_gap_in_the_heartbeat(clock):
    loop, dt = armed(clock, mode="path")
    run(loop, clock, dt, 25.0)
    step, s, laps = loop.step, loop.step.s, loop.step.laps
    assert laps >= 1
    run(loop, clock, dt, 0.5, keys=False)
    assert loop.stale and loop.step is step and step.s >= s and step.laps == laps
    run(loop, clock, dt, 25.0)
    square = loop.cfg["control"]["path_mm"]
    view = loop.snapshot()["path"]
    assert view["points"] == square and view["laps"] == step.laps > laps and not view["done"]


@pytest.mark.parametrize("mode", ["balance", "hand"])
def test_modes_that_cannot_resume_are_rebuilt_over_a_gap_in_the_heartbeat(clock, mode):
    loop, dt = armed(clock, mode=mode, guard_on=True)
    run(loop, clock, dt, 0.1)
    step = loop.step
    run(loop, clock, dt, 0.5, keys=False)
    assert loop.stale and loop.step is not step and type(loop.step) is type(step)
    assert loop.snapshot()["path"] is None
