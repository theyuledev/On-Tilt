import math

import numpy as np
import pytest

from src.vision.line_graph import Chain, longest_walk
from src.vision.line_path import seam
from src.vision.line_vote import agree, closes
from src.vision.maze_path import length
from tests.unit.line_pictures import VIS, along, ball_px, follows, lamps, picture, run

pytestmark = pytest.mark.unit

GLARE = [(100.0, 100.0), (-110.0, -100.0)]


def ring(cx, cy, r, n=90):
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n)]


def read(lines, ball=None, r_mm=8.0):
    img = lamps(picture(VIS, lines, 4, ball=ball and (ball, r_mm)), VIS, GLARE)
    return run(img, ball=ball and ball_px(VIS, ball, r_mm))


CIRCLE = ring(0.0, 0.0, 50.0)
# one stroke crossing itself at the middle at 90 degrees
EIGHT = [
    (
        110 * math.cos(t) / (1 + math.sin(t) ** 2),
        110 * math.sin(t) * math.cos(t) / (1 + math.sin(t) ** 2),
    )
    for t in np.linspace(0, 2 * math.pi, 200)
]
LOOP = ring(0.0, 0.0, 60.0)
# two tails, 59 and 47 mm, running into the loop at 30 degrees, at its left and right
TAILS = [
    [(-60.0 - 59 * 0.5, 59 * 0.866), (-60.0, 0.0)],
    [(60.0 + 47 * 0.5, -47 * 0.866), (60.0, 0.0)],
]


def lap(got):
    k = closes(got.points)
    assert got.reason == "" and k is not None
    return k, got.points[k:]


def test_a_circle_closes_on_its_first_point_which_is_its_lowest_x_plus_y():
    k, pts = lap(read([CIRCLE]))
    assert k == 0 and along(pts, CIRCLE)[0].max() < 2
    assert abs(length(pts) - 2 * math.pi * 50) < 10
    assert math.dist(pts[0], (-50 / math.sqrt(2), -50 / math.sqrt(2))) < VIS["line_step_mm"]


def test_a_ball_on_a_loop_puts_the_seam_under_the_ball():
    k, pts = lap(read([CIRCLE], ball=(50.0, 0.0)))
    assert k == 0 and math.dist(pts[0], (50.0, 0.0)) < 1
    assert along(pts[1:-1], CIRCLE)[0].max() < 2


def test_a_figure_eight_goes_straight_over_its_crossing_and_closes():
    k, pts = lap(read([EIGHT]))
    assert along(pts, EIGHT)[0].max() < 2
    assert abs(length(pts) - length(EIGHT)) < 15


@pytest.mark.parametrize("ball", [None, TAILS[0][0]])
def test_a_tail_into_a_loop_is_run_once_then_the_loop_is_lapped(ball):
    got = read([TAILS[0], LOOP], ball=ball)
    k, pts = lap(got)
    # the seam is where the tail's stroke meets the loop's, a few mm up the tail at 30 degrees
    assert math.dist(got.points[0], TAILS[0][0]) < 5 and math.dist(pts[0], TAILS[0][1]) < 10
    assert along(got.points[:k], TAILS[0])[0].max() < 2 and along(pts[1:-1], LOOP)[0].max() < 2
    assert abs(length(pts) - 2 * math.pi * 60) < 10 and got.skipped == []


def test_of_two_tails_the_longer_is_taken_every_time_and_the_other_skipped():
    img = lamps(picture(VIS, TAILS + [LOOP], 4), VIS, GLARE)
    got = run(img)
    k, pts = lap(got)
    assert math.dist(got.points[0], TAILS[0][0]) < 5 and math.dist(pts[0], TAILS[0][1]) < 10
    # drawn from the route: its last point is where the tail's stroke meets the loop's
    assert len(got.skipped) == 1 and along(got.skipped[0][:-1], TAILS[1])[0].max() < 3
    assert run(img) == got


def test_two_loops_through_one_point_are_lapped_both_after_the_stem():
    petals = [ring(-35.0, 35.0, 35 * math.sqrt(2)), ring(35.0, 35.0, 35 * math.sqrt(2))]
    stem = [(0.0, -100.0), (0.0, 0.0)]
    got = read([stem] + petals)
    k, pts = lap(got)
    assert math.dist(got.points[0], stem[0]) < 5 and math.dist(pts[0], stem[1]) < 10
    assert along(got.points[:k], stem)[0].max() < 2
    assert all(along(p, pts)[0].max() < 3 for p in petals)


@pytest.mark.parametrize("around, lapped", [(112.0, True), (88.0, False)])
def test_an_end_loop_is_lapped_from_line_loop_min_mm_round(around, lapped):
    r = around / (2 * math.pi)
    got = read([[(-120.0, 0.0), (0.0, 0.0)], ring(r, 0.0, r)])
    assert got.reason == "" and (closes(got.points) is not None) == lapped


def test_a_c_with_the_ball_in_its_30_mm_gap_stays_open():
    c = [(50 * math.cos(t), 50 * math.sin(t)) for t in np.linspace(0.3, 2 * math.pi - 0.3, 80)]
    got = read([c], ball=(50.0, 0.0))
    assert got.reason == "" and closes(got.points) is None and along(got.points, c)[0].max() < 2


@pytest.mark.parametrize("long", [41.5, 60.0])
def test_a_branch_left_out_is_measured_and_drawn_from_the_route(long):
    bar, stem = [(-90.0, 0.0), (90.0, 0.0)], [(0.0, 0.0), (0.0, long)]
    got = read([bar, stem])
    follows(got, bar)
    assert len(got.skipped) == 1 and min(abs(y) for _, y in got.skipped[0]) < 1


def test_a_t_follows_the_bar_and_skips_the_stem():
    bar, stem = [(-90.0, 0.0), (90.0, 0.0)], [(0.0, 0.0), (0.0, 70.0)]
    got = read([bar, stem])
    follows(got, bar)
    assert len(got.skipped) == 1 and along(got.skipped[0], stem)[0].max() < 2
    assert agree([got] * 5, VIS) == got


def test_a_cross_left_out_comes_back_as_two_straight_strokes():
    cross = [[(-40.0, 60.0), (40.0, 140.0)], [(-40.0, 140.0), (40.0, 60.0)]]
    got = read([LONG] + cross)
    follows(got, LONG)
    assert sorted(min(along(s, c)[0].max() for c in cross) < 2 for s in got.skipped) == [True] * 2


def chain(pts, nodes):
    px = np.array(pts, float)
    far = min(20, len(px) - 1)
    out = [px[0] - px[far], px[-1] - px[-1 - far]]
    steps = float(np.linalg.norm(np.diff(px, axis=0), axis=1).sum())
    return Chain(px, nodes, tuple(o / np.linalg.norm(o) for o in out), steps)


def line(a, b):
    n = round(math.dist(a, b))
    return [(a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n) for t in range(n + 1)]


def test_a_join_from_a_line_end_onto_a_fork_pays_its_length():
    # stem and two arms meet at a fork; a line ends 47 px short of it, in line with one arm
    stem = chain(line((100, 200), (100, 101)), (10, 1))
    arms = [chain(line((101, 99), (170, 30)), (1, 11)), chain(line((99, 99), (30, 30)), (1, 12))]
    short = chain(line((0, 200), (68, 132)), (13, 14))
    found = [stem, *arms, short]
    joined = {(3, 1): [(1, 0, 0)], (1, 0): [(3, 1, 0)]}
    starts = [(i, e) for i in range(4) for e in (0, 1)]
    walk = longest_walk(found, joined, VIS["line_bend_deg"], 25.0, starts)
    assert {i for i, _ in walk} in ({0, 1}, {0, 2})


def test_the_lap_starts_where_the_walk_first_reached_the_knot_it_ends_at():
    # two forks 1 px apart are one knot: the lap is both loops, not only the last
    stem = chain(line((0, 100), (0, 0)), (10, 1))
    loops = [
        chain([(0, 0), (-40, 40), (-80, 0), (-40, -40), (-1, -1)], (1, 1)),
        chain([(1, 1), (40, 40), (80, 0), (40, -40), (3, 3)], (2, 2)),
    ]
    found = [stem, *loops, chain([(0, 0), (1, 1)], (1, 2))]
    assert seam(found, [(0, True), (1, True), (2, True)], 7) == len(stem.px) - 1


LONG, SHORT = [(-100.0, -60.0), (100.0, -60.0)], [(-60.0, 60.0), (60.0, 60.0)]


def test_two_drawings_without_a_ball_give_the_longer_and_skip_the_other():
    got = read([SHORT, LONG])
    follows(got, LONG)
    assert len(got.skipped) == 1 and along(got.skipped[0], SHORT)[0].max() < 2


def test_a_ball_on_the_shorter_drawing_picks_it_and_skips_the_longer():
    got = read([SHORT, LONG], ball=SHORT[0])
    follows(got, SHORT)
    assert math.dist(got.points[0], SHORT[0]) < 1
    assert len(got.skipped) == 1 and along(got.skipped[0], LONG)[0].max() < 2


def test_a_ball_on_a_fork_joins_the_straightest_two_ways_under_it_and_skips_the_third():
    bar, stem = [(-90.0, 0.0), (90.0, 0.0)], [(0.0, 0.0), (0.0, 70.0)]
    got = read([bar, stem], ball=(0.0, 0.0))
    follows(got, bar)
    assert len(got.skipped) == 1 and along(got.skipped[0], stem)[0].max() < 2
