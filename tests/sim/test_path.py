import math
import time

import pytest

from src.app import Loop
from src.io.settings import load
from src.modes import MODES
from src.modes.path import project

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


def off_path(p, path):
    feet = []
    for a, b in zip(path, path[1:], strict=False):
        s, length = project(p, a, b), math.dist(a, b)
        feet.append((a[0] + s / length * (b[0] - a[0]), a[1] + s / length * (b[1] - a[1])))
    return min(math.dist(p, q) for q in feet)


@pytest.mark.parametrize("kp_scale", [1.0, 1.2])
def test_path_mode_rolls_each_side_at_path_speed_whatever_the_kp(clock, kp_scale):
    cfg = load()
    c = cfg["control"]
    c["mode"] = "path"
    c["kp_deg_mm"] *= kp_scale
    path = [tuple(p) for p in c["path_mm"]]
    balls = list(follow(cfg, clock, 20.0))
    for a, b in zip(path[1:], path[2:], strict=False):
        length = math.dist(a, b)
        mid = []
        for ball in balls:
            p = (ball.x_mm, ball.y_mm)
            s = project(p, a, b)
            if length / 4 <= s <= 3 * length / 4 and off_path(p, [a, b]) < 30:
                mid.append((ball.t_s, s))
        speed = (mid[-1][1] - mid[0][1]) / (mid[-1][0] - mid[0][0])
        assert speed == pytest.approx(c["path_speed_mm_s"], rel=0.1)


def test_path_mode_keeps_to_the_line_and_the_corners_and_well_off_the_rim(clock):
    cfg = load()
    c, rig = cfg["control"], cfg["rig"]
    c["mode"] = "path"
    edge = (rig["plate_inner_mm"] - rig["ball_d_mm"]) / 2
    path = [tuple(p) for p in c["path_mm"]]
    pts = [(ball.x_mm, ball.y_mm) for ball in follow(cfg, clock, 20.0)]
    # a feed forward with the wrong sign throws the ball wide at every corner
    assert min(edge - max(abs(x), abs(y)) for x, y in pts) >= 10
    corners = path[1:-1]
    for corner in corners:
        assert min(math.dist(p, corner) for p in pts) < 10
    joined = next(i for i, p in enumerate(pts) if math.dist(p, corners[0]) < 10)
    for p in pts[joined:]:
        if min(math.dist(p, q) for q in path) > 30:
            assert off_path(p, path) < c["path_capture_mm"]


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
    loop.step.seg = 2
    loop.arm()
    assert loop.mode == "path" and loop.step.seg == 0
