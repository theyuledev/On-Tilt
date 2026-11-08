import math
import time

import pytest

from src.app import Loop
from src.io.settings import load
from src.modes import MODES

pytestmark = pytest.mark.sim


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def follow(cfg, clock, seconds):
    loop = Loop(cfg)
    loop.arm()
    dt = 1 / cfg["control"]["sim_tick_hz"]
    for _ in range(round(seconds / dt)):
        clock[0] += dt
        # a live but idle intent, otherwise the stale guard levels the plate
        loop.keys.set(0.0, 0.0)
        yield loop.tick(clock[0], dt).ball


def test_path_mode_laps_the_square_corner_by_corner_without_touching_the_rim(clock):
    cfg = load()
    cfg["control"]["mode"] = "path"
    rig = cfg["rig"]
    edge = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2
    corners = [tuple(p) for p in cfg["control"]["path_mm"][1:]]
    for ball in follow(cfg, clock, 60.0):
        assert abs(ball.x_mm) < edge and abs(ball.y_mm) < edge
        if corners and math.dist((ball.x_mm, ball.y_mm), corners[0]) < rig["ball_d_mm"] / 2:
            corners.pop(0)
    assert corners == []
    end = cfg["control"]["path_mm"][-1]
    assert math.dist((ball.x_mm, ball.y_mm), end) < cfg["control"]["path_capture_mm"]


def test_a_one_point_path_parks_the_ball_on_the_point(clock):
    cfg = load()
    cfg["control"]["mode"] = "path"
    cfg["control"]["path_mm"] = [[50, -30]]
    *_, ball = follow(cfg, clock, 15.0)
    assert math.dist((ball.x_mm, ball.y_mm), (50, -30)) < cfg["control"]["path_capture_mm"]


def test_mode_is_picked_by_name_and_arming_again_starts_the_path_over():
    loop = Loop(load())
    assert loop.step.mode is MODES["hand"]
    loop.mode = "path"
    loop.step.seg = 2
    loop.arm()
    assert loop.mode == "path" and loop.step.seg == 0
