import math
import time

import numpy as np
import pytest

from src.vision.line_path import NONE
from src.vision.line_vote import DIFFER, LinePath, agree, closes
from tests.unit.line_pictures import STRAIGHT, VIS, along, follows, picture, run

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["black", "white", "noise"])
def test_a_blank_or_noise_frame_is_refused_fast(kind):
    img = {
        "black": np.zeros((800, 1280), np.uint8),
        "white": np.full((800, 1280), 255, np.uint8),
        "noise": np.random.default_rng(0).integers(0, 256, (800, 1280), dtype=np.uint8),
    }[kind]
    start = time.perf_counter()
    assert run(img).reason == "no line found"
    assert time.perf_counter() - start < 1


def test_five_reads_give_the_path_most_agree_on():
    full, short = picture(VIS, [STRAIGHT]), picture(VIS, [[STRAIGHT[0], (0.0, 10.0)]])
    reads = [run(full)] * 3 + [run(short), run(picture(VIS, []))]
    assert reads[3].reason == "" and reads[4].reason == "no line found"
    follows(agree(reads, VIS), STRAIGHT)


def test_three_reads_that_stop_early_do_not_outvote_two_that_go_on():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    early = LinePath(whole.points[:21] + [[5.0, 8.0], [8.0, 17.0]], "")
    assert agree([early] * 3 + [whole] * 2, VIS) == LinePath([], DIFFER)
    assert agree([whole] * 3 + [early] * 2, VIS) == whole
    for short, refused in ((6, False), (7, True)):
        stop = LinePath(whole.points[:-short], "")
        assert (agree([stop] * 3 + [whole] * 2, VIS).reason == DIFFER) == refused
    elsewhere = LinePath([[float(x), 60.0] for x in range(-200, 201, 10)], "")
    assert agree([whole] * 3 + [elsewhere] * 2, VIS) == whole


def test_one_read_of_four_that_goes_on_does_not_refuse_the_rest():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    early = LinePath(whole.points[:21] + [[5.0, 8.0], [8.0, 17.0]], "")
    assert agree([early] * 3 + [whole], VIS) == early
    assert agree([early] * 2 + [whole] + [early], VIS) == early


def test_two_reads_that_stop_early_do_not_beat_one_that_goes_on():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    early = LinePath(whole.points[:21] + [[5.0, 8.0], [8.0, 17.0]], "")
    many = LinePath([], NONE)
    assert agree([early, many, whole, early, many], VIS) == LinePath([], DIFFER)


def test_two_reads_on_the_same_wrong_way_still_beat_one_right_read():
    right = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    wrong = LinePath([[x, 30.0 if abs(x) <= 20 else 0.0] for x, _ in right.points], "")
    close = LinePath([[x, y + 2.0 * (y > 0)] for x, y in wrong.points], "")
    many = LinePath([], NONE)
    got = agree([wrong, right, close, many, many], VIS)
    assert got in (wrong, close)


def test_paths_that_outnumber_the_refusals_win_when_most_of_them_agree():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    detour = LinePath([[x, 30.0 if abs(x) <= 20 else 0.0] for x, _ in whole.points], "")
    many = LinePath([], NONE)
    assert agree([whole, many, detour, many, whole], VIS) == whole
    assert agree([whole, whole, detour, many], VIS) == whole
    assert agree([whole, whole, many, many], VIS) == LinePath([], DIFFER)
    assert agree([whole, detour, many, many, many], VIS) == LinePath([], NONE)
    assert agree([whole, detour, LinePath([], NONE), whole, detour], VIS) == LinePath([], DIFFER)


def test_of_reads_that_agree_the_one_whose_ends_match_the_rest_is_taken():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    tail = LinePath(whole.points + [[200.0, 16.0]], "")
    other = LinePath([[-200.0, -16.0]] + whole.points, "")
    assert agree([tail, whole, other, whole, whole], VIS) == whole


def test_the_pick_is_one_of_the_reads_most_agree_on():
    xs = range(-200, 201, 10)
    ends = [LinePath([[float(x), dy if abs(x) >= 170 else 0.0] for x in xs], "") for dy in (30, 20)]
    ends += [
        LinePath([[float(x), dy if abs(x) >= 170 else 0.0] for x in xs], "") for dy in (-20, -30)
    ]
    middle = LinePath([[float(x), 6.0 if abs(x) <= 20 else 0.0] for x in xs], "")
    assert agree([ends[0], ends[1], middle, ends[2], ends[3]], VIS) in (ends[1], ends[2])


def test_of_two_that_agree_the_one_closer_to_the_third_in_the_middle_is_taken():
    xs = range(-200, 201, 10)
    bump = [LinePath([[float(x), h if abs(x) <= 20 else 0.0] for x in xs], "") for h in (0, 4, 10)]
    assert agree(bump, VIS) == bump[1]


def test_reads_whose_two_ends_both_wander_still_win_the_vote():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    wander = LinePath([[x, 30.0 if abs(x) >= 170 else 0.0] for x, _ in whole.points], "")
    assert agree([whole] * 3 + [wander] * 2, VIS) == whole


def test_reads_that_do_not_agree_are_refused():
    other = [(-90.0, 60.0), (80.0, 60.0)]
    reads = [run(picture(VIS, [STRAIGHT]))] * 2 + [run(picture(VIS, [other]))] * 2
    assert agree(reads + [run(picture(VIS, []))], VIS) == LinePath([], DIFFER)
    blank = run(picture(VIS, []))
    assert agree([blank] * 3 + reads[:2], VIS) == LinePath([], "no line found")


def test_two_reads_whose_middle_differs_by_a_spiral_turn_do_not_agree():
    base = [[x, 0.0] for x in range(-100, 101, 10)]
    moved = [[x, 17.0 if -10 <= x <= 10 else 0.0] for x, _ in base]
    assert agree([LinePath(base, ""), LinePath(moved, "")], VIS).reason == DIFFER


def test_reads_whose_ends_wander_still_agree():
    base = [[x, 0.0] for x in range(-100, 101, 10)]
    wander = [[x, 30.0 if abs(x) == 100 else 0.0] for x, _ in base]
    assert agree([LinePath(base, ""), LinePath(wander, "")], VIS).reason == ""


def test_a_crowded_board_gives_the_same_route_every_time_without_hanging():
    # the walk search runs out of steps here; what it found by then stands
    grid = [[(k, -120.0), (k, 120.0)] for k in range(-120, 121, 40)]
    grid += [[(-120.0, k), (120.0, k)] for k in range(-120, 121, 40)]
    img = picture(VIS, grid, 4)
    t0 = time.perf_counter()
    got = run(img)
    assert time.perf_counter() - t0 < 10
    assert got.reason == "" and got.skipped and run(img) == got
    assert np.min([along(got.points, g)[0] for g in grid], 0).max() < 2


RING = [
    [round(60 * math.cos(a), 1), round(60 * math.sin(a), 1)]
    for a in np.linspace(0, 2 * math.pi, 38)[:-1]
]


def ring(start):
    pts = RING[start:] + RING[:start]
    return LinePath(pts + pts[:1], "")


def test_closing_and_open_reads_of_one_loop_do_not_agree():
    shut, open_ = ring(0), LinePath(RING, "")
    assert closes(shut.points) == 0 and closes(open_.points) is None
    assert agree([shut] * 3 + [open_] * 2, VIS) == shut
    assert agree([shut] * 2 + [open_] * 3, VIS) == open_


def test_loops_read_from_different_seams_agree_but_not_ones_that_differ_before_the_close():
    got = agree([ring(0), ring(9), ring(0), ring(20), ring(9)], VIS)
    assert got.reason == "" and closes(got.points) == 0
    bent = ring(0).points
    bent[-4:-1] = [[x * 1.4, y * 1.4] for x, y in bent[-4:-1]]
    assert agree([ring(0)] * 2 + [LinePath(bent, "")] * 2, VIS) == LinePath([], DIFFER)


def test_a_loop_with_a_different_tail_does_not_agree_and_a_split_is_refused():
    tails = [
        [[-60.0 - d, -d] for d in range(50, 0, -10)],
        [[-60.0 - d, d] for d in range(50, 0, -10)],
    ]
    a, b = (LinePath(t + RING[18:] + RING[:19], "") for t in tails)
    assert closes(a.points) == 5 and agree([a, a, b, a, b], VIS) == a
    assert agree([a, b, a, b, LinePath([], NONE)], VIS) == LinePath([], DIFFER)


def test_reads_that_left_out_much_ink_sit_out_while_the_sure_ones_agree():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    detour = LinePath([[x, 30.0 if abs(x) <= 20 else 0.0] for x, _ in whole.points], "")
    cut = [LinePath(whole.points[:k], "", [whole.points[k:]]) for k in (20, 24)]
    assert agree([whole, detour, cut[0], whole, cut[1]], VIS) == whole
    assert (
        agree([LinePath(p.points, "") for p in cut] + [whole, detour, whole], VIS).reason == DIFFER
    )
    assert agree([cut[0]] * 5, VIS) == cut[0]
