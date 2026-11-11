import pytest

from src.contracts import BallState, Intent
from src.modes import MODES, assist, make_mode
from src.modes.assist import goal_mm

pytestmark = pytest.mark.unit

RIG = {"plate_inner_mm": 305, "ball_d_mm": 61.8}
CFG = {
    "rig": RIG,
    "control": {
        "kp_deg_mm": 0.1,
        "kd_deg_s_mm": 0.01,
        "d_max_deg": 2.0,
        "assist_margin_mm": 20,
        "path_lookahead_mm": 40,
        "unstick_gain": 0.0,
    },
}
REACH = 305 / 2 - 61.8 / 2 - 20


def intent(x, y, valid=True):
    return Intent(x, y, valid=valid, t_s=0.0)


def ball(x, y, vx=0.0, vy=0.0, valid=True):
    return BallState(x, y, vx, vy, valid=valid, t_s=0.0)


def test_goal_sits_at_the_centre_without_a_hand():
    assert goal_mm(intent(1.0, 1.0, valid=False), RIG, 20) == (0.0, 0.0)


def test_full_intent_reaches_the_corner_one_margin_short_of_the_rim():
    assert REACH == pytest.approx(101.6)
    assert goal_mm(intent(1.0, 1.0), RIG, 20) == pytest.approx((REACH, REACH))


def test_goal_follows_each_axis_sign_and_scale():
    assert goal_mm(intent(-0.5, 0.25), RIG, 20) == pytest.approx((-REACH / 2, REACH / 4))


def test_intent_past_one_is_clamped():
    assert goal_mm(intent(2.0, -3.0), RIG, 20) == pytest.approx((REACH, -REACH))


def test_assist_levels_without_a_ball():
    cmd = assist(ball(50, 50, valid=False), intent(1.0, 1.0), CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == (0.0, 0.0)


def test_assist_chases_a_far_goal_one_lookahead_at_a_time():
    cmd = assist(ball(0, 0), intent(1.0, 0.0), CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((0.1 * 40, 0.0))


def test_assist_pulls_straight_onto_a_near_goal():
    cmd = assist(ball(REACH - 10, REACH + 10), intent(1.0, 1.0), CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((1.0, -1.0))


def test_assist_is_a_plain_mode():
    assert make_mode("assist", CFG) is assist
    assert MODES["assist"] is assist
