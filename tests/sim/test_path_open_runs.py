import bisect
import math

import path_rig as rig
import pytest

from src.contracts import BallState, PlateCommand, PlateState
from src.modes import make_mode
from src.modes.assist import reach_mm
from src.modes.path import lengths

pytestmark = pytest.mark.sim


def level(ball, intent, cfg):
    return PlateCommand(0.0, 0.0)


def test_the_rig_ball_sticks_under_the_start_lean_and_rolls_over_it():
    start = rig.STRAIGHT[0]
    held = rig.follow(rig.STRAIGHT, end=start, seconds=1.0, gaps=(), lean=(1.4, 0.0), mode=level)
    rolls = rig.follow(rig.STRAIGHT, end=start, seconds=1.0, gaps=(), lean=(1.6, 0.0), mode=level)
    assert held["end_mm"] == 0.0
    assert rolls["end_mm"] > 20.0 and rolls["end_mm_s"] > 50.0


def test_the_rig_ball_slows_on_a_level_plate_and_stays_where_it_stops():
    cfg, flat = rig.rig_cfg(rig.STRAIGHT), PlateState(0.0, 0.0, 0.0, 0.0)
    for lean in ((0.0, 0.0), (-1.0, 0.0)):
        ball = BallState(0.0, 0.0, 9.0, 0.0, valid=True, t_s=0.0)
        for i in range(1, 101):
            ball = rig.sticky_roll(ball, flat, i * 0.01, cfg["rig"], cfg["control"], lean)
        assert ball.vx_mm_s == 0.0 and 0.0 < ball.x_mm < 2.0


def test_the_mode_sees_the_rig_ball_late_and_not_at_all_in_the_gaps():
    seen = []

    def spy(ball, intent, cfg):
        seen.append(ball)
        return PlateCommand(0.0, 0.0)

    r = rig.follow(rig.STRAIGHT, seconds=3.0, mode=spy)
    servo_s = rig.rig_cfg(rig.STRAIGHT)["control"]["servo_dead_time_s"]
    assert 3.0 - seen[-1].t_s == pytest.approx(rig.SEEN_LATE_S - servo_s, abs=0.011)
    assert sum(not b.valid for b in seen) == r["lost_ticks"]
    assert 39 <= r["lost_ticks"] <= 41


def test_the_rig_reports_how_far_out_the_ball_got():
    r = rig.follow(rig.STRAIGHT, seconds=6.0, gaps=(), lean=(1.6, 0.0), mode=level)
    assert r["far_mm"] == pytest.approx((292.0 - 16.0) / 2)


def test_the_table_has_a_row_for_every_case_and_path():
    rows = rig.table().splitlines()
    assert len(rows) == 1 + len(rig.CASES) * len(rig.PATHS)
    assert rows[1].split()[:2] == ["default", "straight"] and "True" in rows[1]


def test_the_rig_calls_a_level_plate_unfinished_and_counts_the_blind_ticks():
    r = rig.follow(rig.STRAIGHT, seconds=8.0, mode=level)
    assert not r["finished"] and math.isnan(r["time_s"]) and math.isnan(r["done_s"])
    assert 128 <= r["lost_ticks"] <= 132
    assert r["off_max_mm"] == 0.0
    assert r["end_mm"] == pytest.approx(math.dist(rig.STRAIGHT[0], rig.STRAIGHT[-1]))


def test_bursts_leave_about_a_third_of_the_time_blind():
    blind = sum(n for _, n in rig.bursts(60.0))
    assert 0.28 < blind / 60.0 < 0.38
    assert all(0.1 <= n <= 0.3 for _, n in rig.bursts(60.0))


def test_off_line_is_the_distance_to_the_nearest_piece_of_the_path():
    bend = [(-10.0, 0.0), (10.0, 0.0), (10.0, 20.0)]
    assert rig.off_line((0.0, 7.0), bend) == pytest.approx(7.0)
    assert rig.off_line((30.0, 0.0), bend) == pytest.approx(20.0)
    assert rig.off_line((12.0, 10.0), bend) == pytest.approx(2.0)


# case -> finished within s, off the line max and median under mm (measured: default 5.0 to 6.5 s,
# 9.0, 4.1; blind 5.1 to 6.4 s, 9.4, 3.3; rig keys 4.9 to 6.8 s, 12.9, 4.2)
BOUNDS = {
    "default": (10.0, 12.0, 6.0),
    "blind 1/3": (10.0, 13.0, 6.0),
    "rig keys": (12.0, 16.0, 7.0),
}


@pytest.mark.parametrize("case", rig.CASES)
@pytest.mark.parametrize("name", rig.PATHS)
def test_an_open_path_is_followed_to_its_end_where_the_ball_comes_to_rest(case, name):
    within_s, off_max, off_median = BOUNDS[case]
    r = rig.run_case(case, name)
    assert r["finished"] and 3.0 < r["time_s"] < within_s and 0.0 < r["finish_mm_s"] < 60.0
    assert r["off_max_mm"] < off_max and r["off_median_mm"] < off_median
    assert r["end_mm_s"] < 1.0 and r["end_mm"] < 15.0


def test_the_mode_says_done_when_the_ball_really_is_at_the_end():
    r = rig.follow(rig.S_CURVE, gaps=())
    assert abs(r["done_s"] - r["time_s"]) < 0.3


def test_a_ball_by_the_last_point_follows_the_path_the_other_way():
    r = rig.follow(rig.STRAIGHT, start=(95.0, 35.0), end=rig.STRAIGHT[0])
    assert r["finished"] and r["time_s"] < 10.0 and r["off_max_mm"] < 12.0


def test_a_ball_at_rest_beside_the_start_gets_going():
    x, y = rig.STRAIGHT[0]
    r = rig.follow(rig.STRAIGHT, start=(x - 9.3, y + 23.2))
    assert r["finished"] and r["time_s"] < 10.0
    # the distance off the line is largest where it started and small for most of the run
    assert r["off_max_mm"] == pytest.approx(25.0, abs=0.5) and 0.0 < r["off_median_mm"] < 5.0


def test_a_lean_against_the_way_it_goes_does_not_stop_the_ball():
    (x0, y0), (x1, y1) = rig.STRAIGHT[0], rig.STRAIGHT[-1]
    n = math.dist((x0, y0), (x1, y1))
    r = rig.follow(rig.STRAIGHT, lean=(-0.5 * (x1 - x0) / n, -0.5 * (y1 - y0) / n))
    assert r["finished"] and r["time_s"] < 15.0


OUT_AND_BACK = (
    rig.line((40.0, -40.0), (138.0, -40.0))
    + rig.line((138.0, -40.0), (138.0, 60.0))[1:]
    + rig.line((138.0, 60.0), (40.0, 60.0))[1:]
)


@pytest.mark.parametrize(
    "path, start, end",
    [
        (OUT_AND_BACK, None, (40.0, 60.0)),
        (rig.line((138.0, 20.0), (138.0, -138.0)), (0.0, 0.0), (1, -1)),
    ],
    ids=["out past the reach and back", "along the reach from the centre"],
)
def test_a_path_past_the_reach_is_followed_pulled_in_and_the_ball_stays_off_the_wall(
    path, start, end
):
    cfg = rig.rig_cfg(path)
    reach = reach_mm(cfg["rig"], cfg["control"]["assist_margin_mm"])
    rim = (cfg["rig"]["plate_inner_mm"] - cfg["rig"]["ball_d_mm"]) / 2
    # an end given as 1 or -1 lies on the reach line
    end = tuple(v * reach if abs(v) == 1 else v for v in end)
    r = rig.follow(path, start=start, end=end, cfg=cfg)
    assert r["finished"] and r["far_mm"] < rim - cfg["control"]["guard_margin_mm"]


def test_a_spiral_with_17_mm_between_turns_is_followed_turn_by_turn_through_blind_gaps():
    points, angles = rig.spiral()
    cum, cfg = lengths(points), rig.rig_cfg(points)
    mode, places, track = make_mode("path", cfg), [], []

    def watched(ball, intent, cfg):
        cmd = mode(ball, intent, cfg)
        places.append(mode.open.s)
        return cmd

    gaps = ((10.0, 0.5), (25.0, 0.5))
    r = rig.follow(points, seconds=90.0, gaps=gaps, cfg=cfg, mode=watched, track=track)
    assert r["finished"] and r["time_s"] < 45.0
    assert all(b >= a for a, b in zip(places, places[1:], strict=False))
    apart, off, own = 0.0, 0.0, angles[0]
    for (x, y), s in zip(track, places, strict=True):
        a = math.atan2(y, x)
        # the angle the ball really is at, counted on through the turns
        own = a + 2 * math.pi * round((own - a) / (2 * math.pi))
        i = min(bisect.bisect_right(cum, s), len(cum) - 1)
        f = (s - cum[i - 1]) / (cum[i] - cum[i - 1])
        apart = max(apart, abs(angles[i - 1] + f * (angles[i] - angles[i - 1]) - own))
        off = max(off, abs(math.hypot(x, y) - rig.spiral_r(own)))
    # measured: 1.6 m in 33.3 s, never over 0.06 of a turn from its place, 9.4 mm off its own turn
    assert apart < 0.25 * 2 * math.pi and off < 12.0
