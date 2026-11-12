from collections import Counter

import pytest

from src.vision.bench import report
from src.vision.bench.replay import MOVING, WARMING, Scored
from src.vision.bench.score import (
    ALL,
    CORNERS,
    CORRECT,
    EDGE,
    LOST,
    MIDDLE,
    OFF,
    SIDES,
    WRONG,
    Mark,
    judge,
    regions,
)
from src.vision.bench.truth import ARRIVAL, UNSEEN, Label

pytestmark = pytest.mark.unit

R = 20.0
HALF, BAND = 150.0, 15.0


def label(a=(100.0, 100.0), b=None) -> Label:
    return Label(a, b or a, R, ARRIVAL, True, None)


def test_detection_inside_the_ball_is_correct():
    assert judge((100.0, 100.0), label()) == CORRECT
    # the rim still counts, a hair past it does not
    assert judge((100.0 + R, 100.0), label()) == CORRECT
    assert judge((100.0 + R + 0.01, 100.0), label()) == WRONG
    # a moving ball is somewhere on its way between the two snapshots
    rolling = label(b=(200.0, 100.0))
    assert judge((150.0, 100.0 + R), rolling) == CORRECT
    assert judge((200.0 + R, 100.0), rolling) == CORRECT


def test_planted_false_lock_is_called_wrong():
    assert judge((160.0, 100.0), label()) == WRONG
    assert judge((150.0, 100.0 + R + 1), label(b=(200.0, 100.0))) == WRONG
    assert judge((200.0 + R + 1, 100.0), label(b=(200.0, 100.0))) == WRONG


def test_no_detection_on_a_labelled_frame_is_lost():
    assert judge(None, label()) == LOST


def test_regions_split_middle_sides_corners_and_edge_band():
    def at(x: float, y: float) -> tuple[str, ...]:
        return regions(x, y, HALF, BAND)

    assert at(0, 0) == at(50, -50) == at(-50, 50) == (MIDDLE, ALL)
    assert at(50.1, 0) == at(0, -50.1) == at(-134.9, 50) == (SIDES, ALL)
    assert at(50.1, 50.1) == at(-134.9, 134.9) == at(134.9, -51) == (CORNERS, ALL)
    # the band lies on top of the side or corner it runs through
    assert at(135, 0) == at(0, -150) == (SIDES, EDGE, ALL)
    assert at(-135, 60) == at(150, -150) == (CORNERS, EDGE, ALL)
    assert at(150.1, 0) == at(0, -150.1) == at(200, 200) == (OFF, ALL)


def scored() -> Scored:
    cold = [
        Mark((MIDDLE, ALL), CORRECT, 0.5),
        Mark((MIDDLE, ALL), CORRECT, None),
        Mark((SIDES, EDGE, ALL), CORRECT, 1.5),
        Mark((SIDES, EDGE, ALL), WRONG, None),
        Mark((CORNERS, ALL), LOST, None),
        Mark((CORNERS, EDGE, ALL), LOST, None),
        Mark((CORNERS, EDGE, ALL), CORRECT, 4.5),
        Mark((OFF, ALL), LOST, None),
    ]
    locked = [Mark((MIDDLE, ALL), CORRECT, 0.25), Mark((SIDES, EDGE, ALL), WRONG, None)]
    return Scored(13, Counter({UNSEEN: 3, WARMING: 2}), cold, locked, Counter({MOVING: 6}))


def test_the_table_counts_every_region_and_the_edge_band_on_top():
    head = (
        "    region        frames       correct    wrong lock          lost"
        "  off at rest mm (median, p95)"
    )
    assert report.table("one", scored()).splitlines() == [
        "one: 8 of 13 frames scored",
        f"  skipped 5 of 13 frames: 3 {UNSEEN}, 2 {WARMING}",
        "  found from nothing",
        head,
        "    middle             2      2 100.0%      0   0.0%      0   0.0%  0.50, 0.50",
        "    sides              2      1  50.0%      1  50.0%      0   0.0%  1.50, 1.50",
        "    corners            3      1  33.3%      0   0.0%      2  66.7%  4.50, 4.50",
        "    off the board      1      0   0.0%      0   0.0%      1 100.0%  -",
        "    edge band          4      2  50.0%      1  25.0%      1  25.0%  3.00, 4.35",
        "    all                8      4  50.0%      1  12.5%      3  37.5%  1.50, 4.20",
        f"  one step locked on a ball at rest, not run on 6 scored frames: 6 {MOVING}",
        head,
        "    middle             1      1 100.0%      0   0.0%      0   0.0%  0.25, 0.25",
        "    sides              1      0   0.0%      1 100.0%      0   0.0%  -",
        "    corners            0      0      -      0      -      0      -  -",
        "    off the board      0      0      -      0      -      0      -  -",
        "    edge band          1      0   0.0%      1 100.0%      0   0.0%  -",
        "    all                2      1  50.0%      1  50.0%      0   0.0%  0.25, 0.25",
    ]


def test_the_skipped_line_is_always_printed_with_reasons():
    nothing = Scored(0, Counter(), [], [], Counter())
    lines = report.table("two", nothing).splitlines()
    assert lines[:2] == ["two: 0 of 0 frames scored", "  skipped 0 of 0 frames"]
    assert lines[10] == "  one step locked on a ball at rest, not run on 0 scored frames"
    one = Scored(4, Counter({WARMING: 4}), [], [], Counter())
    assert report.table("two", one).splitlines()[1] == f"  skipped 4 of 4 frames: 4 {WARMING}"
