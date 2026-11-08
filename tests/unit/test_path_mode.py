import math

import pytest

from src.contracts import BallState, Intent
from src.modes import MODES, PathFollower, balance, make_mode, pd_toward
from src.modes.path import advance, clip, project, target, walk

pytestmark = pytest.mark.unit

SQ = [(100, 100), (-100, 100), (-100, -100), (100, -100), (100, 100)]
CFG = {
    "control": {
        "kp_deg_mm": 0.1,
        "kd_deg_s_mm": 0.01,
        "d_max_deg": 2.0,
        "path_mm": SQ,
        "path_lookahead_mm": 40,
        "path_capture_mm": 15,
    }
}
NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)


def ball(x, y, vx=0.0, vy=0.0, valid=True):
    return BallState(x, y, vx, vy, valid=valid, t_s=0.0)


def test_project_is_the_distance_along_the_segment_clamped_to_it():
    assert project((50, 7), (0, 0), (100, 0)) == 50
    assert project((-10, 3), (0, 0), (100, 0)) == 0
    assert project((130, -3), (0, 0), (100, 0)) == 100
    assert project((5, 5), (1, 1), (1, 1)) == 0


def test_walk_stays_on_the_segment_wraps_corners_and_holds_the_end():
    assert walk(SQ, 0, 50, 40) == pytest.approx((10, 100))
    assert walk(SQ, 0, 180, 40) == pytest.approx((-100, 80))
    assert walk(SQ, 3, 190, 40) == (100, 100)


def test_advance_waits_for_the_capture_band_then_steps_once_per_corner():
    assert advance(SQ, 0, (0, 90), 15) == 0
    assert advance(SQ, 0, (-84, 95), 15) == 0
    assert advance(SQ, 0, (-86, 95), 15) == 1
    assert advance(SQ, 0, (-110, 95), 15) == 1
    assert advance(SQ, 1, (-110, 95), 15) == 1


def test_advance_skips_short_segments_but_never_leaves_the_last_one():
    short = [(0, 0), (100, 0), (105, 0), (110, 0), (200, 0)]
    assert advance(short, 0, (99, 0), 15) == 3
    assert advance(SQ, 3, (100, 100), 15) == 3


def test_clip_pulls_a_far_goal_to_within_reach():
    assert clip((0, 0), (30, 0), 40) == (30, 0)
    assert clip((0, 0), (0, 100), 40) == pytest.approx((0, 40))
    assert math.dist((10, 10), clip((10, 10), (-90, 70), 40)) == pytest.approx(40)


def test_target_from_the_centre_heads_for_the_lookahead_point_on_the_first_side():
    seg, goal = target(SQ, 0, (0, 0), 40, 15)
    assert seg == 0
    assert math.dist((0, 0), goal) == pytest.approx(40)
    assert goal[0] / goal[1] == pytest.approx(-40 / 100)


def test_target_holds_a_single_point_path():
    assert target([(10, 20)], 0, (10, 20), 40, 15) == (0, (10, 20))
    assert target([(10, 20)], 0, (10, 120), 40, 15) == (0, pytest.approx((10, 80)))


def test_path_follower_refuses_an_empty_path():
    with pytest.raises(ValueError):
        make_mode("path", {"control": {**CFG["control"], "path_mm": []}})


def test_pd_toward_acts_on_the_error_to_the_goal():
    cmd = pd_toward(ball(10, 0, 0, 100), (10, 20), CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((0.0, 1.0))


def test_path_follower_remembers_the_corner_it_passed():
    follow = make_mode("path", CFG)
    assert isinstance(follow, PathFollower)
    follow(ball(-86, 95), NO_INTENT, CFG)
    assert follow.seg == 1
    # on the second side the goal sits 40 mm further down, so the plate tips toward -y
    cmd = follow(ball(-100, 50), NO_INTENT, CFG)
    assert cmd.tilt_x_deg == pytest.approx(0.0)
    assert cmd.tilt_y_deg == pytest.approx(-4.0)
    assert follow.seg == 1


def test_path_follower_parks_on_the_last_point_and_levels_without_a_ball():
    follow = make_mode("path", CFG)
    follow.seg = 3
    cmd = follow(ball(100, 100), NO_INTENT, CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == (0.0, 0.0)
    cmd = follow(ball(0, 0, valid=False), NO_INTENT, CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == (0.0, 0.0)


def test_each_run_gets_its_own_follower_and_plain_modes_come_back_as_is():
    assert make_mode("path", CFG) is not make_mode("path", CFG)
    assert make_mode("balance", CFG) is balance
    assert set(MODES) == {"hand", "balance", "path"}
