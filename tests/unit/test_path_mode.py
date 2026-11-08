import math

import pytest

from src.contracts import BallState, Intent
from src.modes import MODES, PathFollower, balance, make_mode, pd_toward
from src.modes.path import advance, clip, corner_speed, guide, pace, project

pytestmark = pytest.mark.unit

SQ = [(100, 100), (-100, 100), (-100, -100), (100, -100), (100, 100)]
CFG = {
    "control": {
        "kp_deg_mm": 0.1,
        "kd_deg_s_mm": 0.01,
        "d_max_deg": 2.0,
        "sim_g_mm_s2": 9810,
        "path_mm": SQ,
        "path_lookahead_mm": 40,
        "path_capture_mm": 15,
        "path_speed_mm_s": 100,
        "path_accel_mm_s2": 250,
    }
}
C = CFG["control"]
NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)


def ball(x, y, vx=0.0, vy=0.0, valid=True):
    return BallState(x, y, vx, vy, valid=valid, t_s=0.0)


def test_project_is_the_distance_along_the_segment_clamped_to_it():
    assert project((50, 7), (0, 0), (100, 0)) == 50
    assert project((-10, 3), (0, 0), (100, 0)) == 0
    assert project((130, -3), (0, 0), (100, 0)) == 100
    assert project((5, 5), (1, 1), (1, 1)) == 0


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


def test_corner_speed_rounds_a_square_corner_and_stops_at_the_ends():
    assert corner_speed(SQ, 1, 15, 250) == pytest.approx(math.sqrt(250 * 15))
    assert corner_speed(SQ, 0, 15, 250) == 0
    assert corner_speed(SQ, 4, 15, 250) == 0
    assert corner_speed([(0, 0), (50, 0), (100, 0)], 1, 15, 250) == math.inf


def test_pace_ramps_up_cruises_then_brakes_to_the_next_corner():
    assert pace(0, 170, 50, 30, 100, 250) == pytest.approx((50, 250))
    assert pace(85, 170, 50, 30, 100, 250) == (100, 0.0)
    assert pace(160, 170, 50, 0, 100, 250) == pytest.approx((math.sqrt(5000), -250))
    assert pace(180, 170, 50, 30, 100, 250) == pytest.approx((30, -250))


def test_guide_on_the_line_mid_side_holds_path_speed_with_no_pull():
    seg, goal, vel, lean = guide(SQ, 0, (0, 100), C)
    assert (seg, goal) == (0, pytest.approx((0, 100)))
    assert vel == pytest.approx((-100, 0))
    assert lean == pytest.approx((0, 0))


def test_guide_pulls_only_across_the_path_and_joins_a_far_ball_slowly():
    _, goal, vel, _ = guide(SQ, 0, (0, 90), C)
    assert goal == pytest.approx((0, 100))
    assert vel == pytest.approx((-100, 0))
    _, goal, vel, lean = guide(SQ, 0, (0, 0), C)
    assert goal == pytest.approx((0, 40))
    assert vel == pytest.approx((0, 0)) and lean == pytest.approx((0, 0))


def test_guide_parks_on_the_end_and_holds_a_single_point_path():
    assert guide(SQ, 3, (100, 90), C) == (3, (100, 100), (0.0, 0.0), (0.0, 0.0))
    assert guide([(10, 20)], 0, (10, 120), C) == (
        0,
        pytest.approx((10, 80)),
        (0.0, 0.0),
        (0.0, 0.0),
    )


def test_path_follower_refuses_an_empty_path():
    with pytest.raises(ValueError):
        make_mode("path", {"control": {**C, "path_mm": []}})


def test_path_follower_drops_repeated_points():
    cfg = {"control": {**C, "path_mm": [[0, 0], [0, 0], [100, 0], [100, 0], [100, 100]]}}
    follow = make_mode("path", cfg)
    assert follow.path == [(0, 0), (100, 0), (100, 100)]
    follow(ball(50, 0), NO_INTENT, cfg)
    follow(ball(90, 0), NO_INTENT, cfg)
    assert follow.seg == 1


def test_pd_toward_acts_on_the_error_to_the_goal():
    cmd = pd_toward(ball(10, 0, 0, 100), (10, 20), CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((0.0, 1.0))


def test_path_follower_pushes_a_resting_ball_along_and_leaves_a_cruising_one():
    follow = make_mode("path", CFG)
    # first side runs toward -x, so pushing along it tips the plate toward -x
    cmd = follow(ball(0, 100), NO_INTENT, CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((-1.0, 0.0))
    cmd = follow(ball(0, 100, -100, 0), NO_INTENT, CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((0.0, 0.0))


def test_path_follower_leans_back_by_the_braking_accel_before_a_corner():
    follow = make_mode("path", CFG)
    # 5 mm before the brake point the path speed is sqrt(61.2^2 + 2 * 250 * 5)
    v = math.sqrt(250 * 15 + 2 * 250 * 5)
    cmd = follow(ball(-80, 100, -v, 0), NO_INTENT, CFG)
    # along the side (-x) the tilt is -A / (5/7 g) in degrees, about -2.04
    assert -cmd.tilt_x_deg == pytest.approx(-250 / 122.3, abs=0.01)
    assert cmd.tilt_y_deg == pytest.approx(0.0)


def test_path_follower_remembers_the_corner_it_passed():
    follow = make_mode("path", CFG)
    assert isinstance(follow, PathFollower)
    follow(ball(-86, 95), NO_INTENT, CFG)
    assert follow.seg == 1
    # on the second side the path runs toward -y, so a ball at rest is pushed that way
    cmd = follow(ball(-100, 50), NO_INTENT, CFG)
    assert cmd.tilt_x_deg == pytest.approx(0.0)
    assert cmd.tilt_y_deg == pytest.approx(-1.0)
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
