import math

import pytest

from src.contracts import BallState, PlateState
from src.io.settings import load
from src.sim.ball import roll

pytestmark = pytest.mark.sim

DT = 0.001
START = BallState(0.0, 0.0, 0.0, 0.0, valid=True, t_s=0.0)


def run(tilt_x, tilt_y, seconds, ball=START):
    cfg = load()
    plate = PlateState(tilt_x, tilt_y, tilt_x, tilt_y)
    states = []
    for _ in range(round(seconds / DT)):
        ball = roll(ball, plate, ball.t_s + DT, cfg["rig"], cfg["control"])
        states.append(ball)
    return states


def test_positive_tilt_rolls_toward_plus_at_five_sevenths_g_less_friction():
    c = load()["control"]
    a = 5 / 7 * c["sim_g_mm_s2"] * math.sin(math.radians(5.0)) - c["sim_friction_mm_s2"]

    px = run(5.0, 0.0, 0.2)[-1]
    py = run(0.0, 5.0, 0.2)[-1]
    assert px.x_mm > 0 and px.y_mm == 0
    assert py.y_mm > 0 and py.x_mm == 0
    assert px.vx_mm_s == pytest.approx(a * 0.2)
    assert py.vy_mm_s == pytest.approx(a * 0.2)

    back = run(-5.0, -5.0, 0.2)[-1]
    assert back.x_mm < 0 and back.y_mm < 0


def test_coasting_ball_on_a_level_plate_stops_after_v_squared_over_two_f():
    f = load()["control"]["sim_friction_mm_s2"]
    v = 80 / math.sqrt(2)
    states = run(0.0, 0.0, 2 * 80 / f, ball=BallState(0.0, 0.0, v, v, valid=True, t_s=0.0))

    end = states[-1]
    assert math.hypot(end.x_mm, end.y_mm) == pytest.approx(80**2 / (2 * f), abs=0.1)
    assert (end.vx_mm_s, end.vy_mm_s) == (0, 0)
    assert all(s.vx_mm_s >= 0 and s.vy_mm_s >= 0 for s in states)


def test_tilt_under_friction_leaves_a_resting_ball_where_it_is():
    c = load()["control"]
    deg = math.degrees(math.asin(0.9 * c["sim_friction_mm_s2"] / (5 / 7 * c["sim_g_mm_s2"])))

    for tilt in ((deg, 0.0), (0.0, -deg), (deg / math.sqrt(2), deg / math.sqrt(2))):
        end = run(*tilt, 1.0)[-1]
        assert (end.x_mm, end.y_mm, end.vx_mm_s, end.vy_mm_s) == (0, 0, 0, 0)


def test_friction_never_flips_a_slow_ball_within_a_step():
    cfg = load()
    slow = BallState(0.0, 0.0, 1.0, -1.0, valid=True, t_s=0.0)

    end = roll(slow, PlateState(0.0, 0.0, 0.0, 0.0), 0.1, cfg["rig"], cfg["control"])

    assert (end.vx_mm_s, end.vy_mm_s) == (0, 0)


def test_ball_stops_at_rim_and_never_leaves():
    rig = load()["rig"]
    wall = rig["plate_inner_mm"] / 2 - rig["ball_d_mm"] / 2

    states = run(12.0, -12.0, 2.0)
    assert all(abs(s.x_mm) <= wall and abs(s.y_mm) <= wall for s in states)
    end = states[-1]
    assert (end.x_mm, end.y_mm, end.vx_mm_s, end.vy_mm_s) == (wall, -wall, 0, 0)

    off = run(-12.0, 12.0, 0.1, ball=end)[-1]
    assert abs(off.x_mm) < wall and abs(off.y_mm) < wall
