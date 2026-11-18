import math

import pytest

from src.contracts import BallState, Intent
from src.io.settings import load
from src.modes import MODES, Balance, PathFollower, make_mode, pd_toward
from src.modes.path import (
    clip,
    from_nearest_end,
    lengths,
    loop_start,
    progress,
    project,
    stretch,
    tidy,
    walk,
)

pytestmark = pytest.mark.unit

SQ = [(100, 100), (-100, 100), (-100, -100), (100, -100), (100, 100)]
ROW = [(float(x), 0.0) for x in range(-100, 101, 10)]
CFG = {
    "control": {
        "kp_deg_mm": 0.1,
        "kd_deg_s_mm": 0.01,
        "d_max_deg": 2.0,
        "path_mm": SQ,
        "path_lookahead_mm": 40,
        "path_capture_mm": 15,
        "guard_on": False,
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


def test_clip_pulls_a_far_goal_to_within_reach():
    assert clip((0, 0), (30, 0), 40) == (30, 0)
    assert clip((0, 0), (0, 100), 40) == pytest.approx((0, 40))
    assert math.dist((10, 10), clip((10, 10), (-90, 70), 40)) == pytest.approx(40)


def test_path_follower_refuses_an_empty_path():
    with pytest.raises(ValueError):
        make_mode("path", {"control": {**CFG["control"], "path_mm": []}})


def test_pd_toward_acts_on_the_error_to_the_goal():
    cmd = pd_toward(ball(10, 0, 0, 100), (10, 20), CFG)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((0.0, 1.0))


def test_each_run_gets_its_own_follower_and_plain_modes_come_back_as_is():
    cfg = load()
    assert make_mode("path", cfg) is not make_mode("path", cfg)
    assert isinstance(make_mode("path", cfg), PathFollower)
    assert make_mode("hand", CFG) is MODES["hand"]
    assert make_mode("balance", CFG) is not make_mode("balance", CFG)
    assert isinstance(make_mode("balance", CFG), Balance)
    assert set(MODES) == {"hand", "balance", "path", "assist"}


def test_progress_is_the_distance_along_the_path_whatever_the_spacing():
    for path in ([ROW[0], ROW[-1]], ROW):
        s = progress(path, lengths(path), (-43.0, 6.0), 30.0, 40.0)
        assert s == pytest.approx(57.0)
        assert walk(path, 0, 0.0, s + 40.0) == pytest.approx((-3.0, 0.0))


def test_progress_never_goes_back_and_stops_at_the_window():
    cum = lengths(ROW)
    assert progress(ROW, cum, (-80.0, 0.0), 50.0, 40.0) == 50.0
    assert progress(ROW, cum, (50.0, 0.0), 50.0, 40.0) == 90.0
    assert progress(ROW, cum, (150.0, 0.0), 190.0, 40.0) == 200.0


def test_progress_stays_on_this_pass_where_the_path_crosses_itself():
    cross = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (50.0, 50.0), (50.0, -50.0)]
    cum = lengths(cross)
    assert progress(cross, cum, (50.0, 1.0), 30.0, 40.0) == pytest.approx(50.0)
    assert progress(cross, cum, (50.0, 1.0), 240.0, 40.0) == pytest.approx(249.0)


def test_stretch_puts_the_goal_reach_away_however_near_or_far_it_is():
    assert stretch((0, 0), (3, 4), 40) == pytest.approx((24, 32))
    assert stretch((0, 0), (300, 400), 40) == pytest.approx((24, 32))
    assert stretch((5, 5), (5, 5), 40) == (5, 5)


def test_lengths_tidy_and_where_a_path_is_lapped_from():
    assert lengths([(0, 0), (3, 4), (3, 10)]) == [0.0, 5.0, 11.0]
    assert tidy([(0, 0), (0, 0), (1, 0), (1, 0), (0, 0)]) == [(0, 0), (1, 0), (0, 0)]
    assert loop_start(ROW) is None and loop_start([(0, 0), (1, 0)]) is None
    assert loop_start([(1, 1)]) is None
    assert loop_start([(0, 0), (1, 0), (0, 0)]) == 0
    # a tail into a loop, and the first of the points the last one repeats
    assert loop_start([(5, 5), (0, 0), (1, 0), (1, 1), (0, 0)]) == 1
    assert loop_start([(0, 0), (1, 0), (0, 0), (0, 1), (0, 0)]) == 0


def test_from_nearest_end_puts_the_end_by_the_ball_first():
    assert from_nearest_end(ROW, (90, 5)) == ROW[::-1]
    assert from_nearest_end(ROW, (-90, 5)) == ROW
    assert from_nearest_end(ROW, (0, 5)) == ROW
