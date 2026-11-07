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


def test_positive_tilt_rolls_toward_plus_at_five_sevenths_g():
    a = 5 / 7 * load()["control"]["sim_g_mm_s2"] * math.sin(math.radians(5.0))

    px = run(5.0, 0.0, 0.2)[-1]
    py = run(0.0, 5.0, 0.2)[-1]
    assert px.x_mm > 0 and px.y_mm == 0
    assert py.y_mm > 0 and py.x_mm == 0
    assert px.vx_mm_s == pytest.approx(a * 0.2)
    assert py.vy_mm_s == pytest.approx(a * 0.2)

    back = run(-5.0, -5.0, 0.2)[-1]
    assert back.x_mm < 0 and back.y_mm < 0


def test_ball_stops_at_rim_and_never_leaves():
    rig = load()["rig"]
    wall = rig["plate_inner_mm"] / 2 - rig["ball_d_mm"] / 2

    states = run(12.0, -12.0, 2.0)
    assert all(abs(s.x_mm) <= wall and abs(s.y_mm) <= wall for s in states)
    end = states[-1]
    assert (end.x_mm, end.y_mm, end.vx_mm_s, end.vy_mm_s) == (wall, -wall, 0, 0)

    off = run(-12.0, 12.0, 0.1, ball=end)[-1]
    assert abs(off.x_mm) < wall and abs(off.y_mm) < wall
