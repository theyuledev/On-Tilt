import math

import pytest

from src.modes import make_mode
from src.modes.path import lengths, loop_start, walk
from tests.unit.test_path_open import NO_INTENT, ball, cfg_for

pytestmark = pytest.mark.unit

SQ = [(50.0, 50.0), (-50.0, 50.0), (-50.0, -50.0), (50.0, -50.0), (50.0, 50.0)]
# a figure eight: both diagonals cross at the middle, the sides join them
BOW = [(-60.0, -60.0), (60.0, 60.0), (60.0, -60.0), (-60.0, 60.0), (-60.0, -60.0)]
# one straight 60 mm piece into a 400 mm loop that it joins at (-50, 0)
TAIL = [(-110.0, 0.0), (-50.0, 0.0), (-50.0, 50.0), (50.0, 50.0), (50.0, -50.0), (-50.0, -50.0)]
TAIL = [*TAIL, (-50.0, 0.0)]


def led(cfg, spots):
    follow = make_mode("path", cfg)
    for x, y in spots:
        follow(ball(x, y), NO_INTENT, cfg)
    return follow


def along(path, d0, d1, step=10.0):
    """Points on a loop from d0 to d1 mm along it, on into the next laps."""
    run = path + path[loop_start(path) + 1 :] * 3
    return [walk(run, 0, 0.0, d0 + step * i) for i in range(int((d1 - d0) / step) + 1)]


def test_a_loop_is_joined_at_its_point_nearest_the_ball_and_shown_as_given():
    cfg = cfg_for(SQ)
    follow = led(cfg, [(10, 80)])
    view = follow.path_view()
    assert view["chase"] == [10, 50] and view["along_mm"] == 40 and follow.phase == "join"
    assert view["points"] == [list(p) for p in SQ] and view["laps"] == 0 and not view["done"]
    # from the start the path runs on a lap and a lookahead past the place
    assert lengths(follow.path)[-1] >= 40 + 400 + cfg["control"]["path_lookahead_mm"]


@pytest.mark.parametrize("path", [SQ, BOW], ids=["square", "figure eight"])
def test_the_place_goes_round_the_seam_and_counts_whole_laps(path):
    cfg, lap = cfg_for(path), lengths(path)[-1]
    follow = make_mode("path", cfg)
    for d, spot in zip(range(0, 2000, 10), along(path, 0.0, 2.3 * lap), strict=False):
        follow(ball(*spot), NO_INTENT, cfg)
        # a crossing of the figure eight never takes the place to the other pass
        assert (follow.s, follow.laps) == (pytest.approx(d % lap, abs=1e-6), int(d // lap))
        assert follow.phase == "follow" and not follow.done
    assert follow.laps == 2 and len(follow.path) < 4 * len(path)


def test_the_point_steered_for_lies_past_the_seam_on_the_next_lap():
    cfg = cfg_for(SQ)
    follow = led(cfg, [(10, -50), (30, -50), (50, -50), (50, -30), (50, -10), (50, 10)])
    assert follow.path_view()["chase"] == [50, 50]
    follow(ball(50, 40), NO_INTENT, cfg)
    # the bend ahead shortens the look to the capture distance, which runs on round the seam
    assert follow.path_view()["chase"] == [45, 50] and follow.laps == 0


def test_a_tail_is_run_once_from_where_the_ball_is_then_the_loop_is_lapped():
    cfg = cfg_for(TAIL)
    follow = led(cfg, [(-100, 25)])
    assert follow.s == pytest.approx(10.0) and follow.path_view()["chase"] == [-100, 0]
    for d, spot in zip(range(10, 1000, 10), along(TAIL, 10.0, 950.0), strict=False):
        follow(ball(*spot), NO_INTENT, cfg)
        # the loop is lapped from (-50, 0) on, the tail is behind for good
        assert follow.laps == max(d - 60, 0) // 400
        assert follow.s == pytest.approx(d if d < 460 else 60 + (d - 60) % 400, abs=1e-6)
    assert follow.laps == 2


def test_a_ball_by_the_loop_skips_the_tail_and_laps_from_where_it_joined():
    cfg = cfg_for(TAIL)
    follow = led(cfg, [(75, 0)])
    assert follow.s == pytest.approx(260.0) and follow.path_view()["chase"] == [50, 0]
    for spot in along(TAIL, 260.0, 480.0):
        follow(ball(*spot), NO_INTENT, cfg)
    # past the end of the points as given the page gets the place a lap back on them
    assert follow.s == pytest.approx(480.0) and follow.path_view()["along_mm"] == 80
    for spot in along(TAIL, 490.0, 670.0):
        follow(ball(*spot), NO_INTENT, cfg)
    assert follow.laps == 1 and follow.s == pytest.approx(270.0)
    assert follow.path_view()["along_mm"] == 270


def test_a_gap_in_the_heartbeat_keeps_the_place_and_the_laps():
    cfg = cfg_for(SQ)
    follow = make_mode("path", cfg)
    for spot in along(SQ, 0.0, 450.0):
        follow(ball(*spot), NO_INTENT, cfg)
    follow.resume()
    follow(ball(0, 50), NO_INTENT, cfg)
    assert (follow.s, follow.laps, follow.phase) == (pytest.approx(50.0), 1, "follow")


def test_a_longer_lookahead_mid_run_gets_more_path_ahead_and_never_a_finish():
    cfg = cfg_for(SQ)
    follow = led(cfg, [(50, 50), (40, 50)])
    cfg["control"]["path_lookahead_mm"] = 900
    cmd = follow(ball(30, 50), NO_INTENT, cfg)
    assert follow.phase == "follow" and lengths(follow.path)[-1] >= 20 + 400 + 900
    assert math.isfinite(cmd.tilt_x_deg) and math.isfinite(cmd.tilt_y_deg)


@pytest.mark.parametrize(
    "path, spots",
    [
        # a loop shorter than the lookahead
        ([(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (0.0, 0.0)], [(0, 0), (5, 0), (10, 0), (7, 4)]),
        # there and back along one stroke
        ([(0.0, 0.0), (60.0, 0.0), (0.0, 0.0)], [(0, 0), (30, 0), (60, 0), (50, 0), (20, 0)]),
        # two points that are the same: a point to park on
        ([(10.0, 10.0), (10.0, 10.0)], [(30, 10), (12, 10), (10, 10)]),
    ],
    ids=["short loop", "there and back", "one point twice"],
)
def test_odd_paths_neither_crash_nor_stall(path, spots):
    cfg = cfg_for(path)
    follow = make_mode("path", cfg)
    for x, y in spots * 3:
        cmd = follow(ball(x, y), NO_INTENT, cfg)
        assert math.isfinite(cmd.tilt_x_deg) and math.isfinite(cmd.tilt_y_deg)
    view = follow.path_view()
    if len(path) > 2:
        assert view["laps"] >= 2 and not view["done"] and len(follow.path) < 60
    else:
        assert view["points"] == [[10, 10]] and view["done"] and view["laps"] == 0
