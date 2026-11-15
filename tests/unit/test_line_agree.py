import time

import numpy as np
import pytest

from src.vision.line_path import DIFFER, LinePath, agree
from tests.unit.line_pictures import STRAIGHT, VIS, follows, picture, run

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
    assert agree([early] * 3 + [whole] * 2, VIS) == ([], DIFFER)
    assert agree([whole] * 3 + [early] * 2, VIS) == whole
    for short, refused in ((5, False), (10, True)):
        stop = LinePath(whole.points[:-short], "")
        assert (agree([stop] * 3 + [whole] * 2, VIS).reason == DIFFER) == refused
    elsewhere = LinePath([[float(x), 60.0] for x in range(-200, 201, 10)], "")
    assert agree([whole] * 3 + [elsewhere] * 2, VIS) == whole


def test_reads_whose_two_ends_both_wander_still_win_the_vote():
    whole = LinePath([[float(x), 0.0] for x in range(-200, 201, 10)], "")
    wander = LinePath([[x, 30.0 if abs(x) >= 170 else 0.0] for x, _ in whole.points], "")
    assert agree([whole] * 3 + [wander] * 2, VIS) == whole


def test_reads_that_do_not_agree_are_refused():
    other = [(-90.0, 60.0), (80.0, 60.0)]
    reads = [run(picture(VIS, [STRAIGHT]))] * 2 + [run(picture(VIS, [other]))] * 2
    assert agree(reads + [run(picture(VIS, []))], VIS) == ([], DIFFER)
    blank = run(picture(VIS, []))
    assert agree([blank] * 3 + reads[:2], VIS) == ([], "no line found")


def test_two_reads_whose_middle_differs_by_a_spiral_turn_do_not_agree():
    base = [[x, 0.0] for x in range(-100, 101, 10)]
    moved = [[x, 17.0 if -10 <= x <= 10 else 0.0] for x, _ in base]
    assert agree([LinePath(base, ""), LinePath(moved, "")], VIS).reason == DIFFER


def test_reads_whose_ends_wander_still_agree():
    base = [[x, 0.0] for x in range(-100, 101, 10)]
    wander = [[x, 30.0 if abs(x) == 100 else 0.0] for x, _ in base]
    assert agree([LinePath(base, ""), LinePath(wander, "")], VIS).reason == ""


def test_a_grid_is_not_a_line():
    grid = [[(k, -120.0), (k, 120.0)] for k in range(-120, 121, 40)]
    grid += [[(-120.0, k), (120.0, k)] for k in range(-120, 121, 40)]
    assert run(picture(VIS, grid, 4)).reason != ""
