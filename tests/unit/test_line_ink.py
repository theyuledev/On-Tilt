import warnings

import cv2
import numpy as np
import pytest

from src.vision.line_ink import marks
from src.vision.px_to_mm import outline_px
from tests.unit.line_pictures import (
    FAINT,
    HALF,
    S_CURVE,
    STRAIGHT,
    TURNED,
    VIS,
    along,
    follows,
    picture,
    run,
)

pytestmark = pytest.mark.unit

WIPED = [(-60.0, -80.0), (60.0, -100.0)]
ALL = np.ones((60, 200), np.uint8)


def board(stroke, *more):
    """Darkness of a board with a long stroke on rows 20-23 and more (rows, cols, darkness)."""
    dark = np.zeros(ALL.shape, np.float32)
    dark[20:24, 10:190] = stroke
    for rows, cols, value in more:
        dark[rows, cols] = value
    return dark


def test_what_a_wipe_left_beside_a_dark_stroke_is_not_ink():
    # the stroke is 17% darker than the board, the wiped line 6%: over line_core, so ink alone
    wiped = picture(VIS, [WIPED], 4, shade=193)
    assert run(wiped).reason == ""
    follows(run(np.minimum(picture(VIS, [S_CURVE], 4, shade=170), wiped)), S_CURVE)


def test_a_faint_tail_on_a_dark_stroke_is_not_ink_but_its_soft_edge_is():
    tail, edge = (slice(24, 40), slice(100, 104), 0.06), (19, slice(10, 190), 0.15)
    found, faint = marks(board(0.4, tail, edge), VIS, ALL)
    assert found[21, 50] and found[19, 50] and not found[30, 101]
    assert faint[30, 101] and not faint[21, 50]


def test_the_limits_never_drop_below_the_fixed_ones():
    # a faint stroke: half its darkness is under line_core and half of line_core under line_weak
    tail, apart = (slice(24, 40), slice(100, 104), 0.028), (slice(45, 49), slice(50, 80), 0.045)
    found = marks(board(0.07, tail, apart), VIS, ALL)[0]
    assert found[21, 50] and not found[30, 101] and not found[46, 60]


def test_a_much_lighter_stretch_after_a_pen_lift_is_left_out_but_comes_back_skipped():
    # ink about 4x lighter than most of the line is not the route, it is shown as left out
    a, b = np.array(STRAIGHT)
    n = np.linalg.norm(b - a)
    lift = [tuple(a + (b - a) * mm / n) for mm in (95, 97)]
    faint = picture(VIS, [[lift[1], STRAIGHT[1]]], 4, shade=FAINT)
    for shade, cover in ((150, n), (120, 95)):
        got = run(np.minimum(picture(VIS, [[STRAIGHT[0], lift[0]]], 4, shade=shade), faint))
        off, arc = along(got.points, STRAIGHT)
        assert got.reason == "" and off.max() < 2
        assert abs(arc.max() - arc.min() - cover) < 5
        rest = [along(s, STRAIGHT) for s in got.skipped]
        assert [round(a.max() - a.min()) // 10 for _, a in rest] == ([] if cover == n else [5])
        assert all(off.max() < 2 and a.min() > cover for off, a in rest)


def test_a_weak_limit_over_the_core_one_keeps_a_clear_line():
    follows(run(picture(VIS, [S_CURVE], 4, shade=170), {**VIS, "line_weak": 0.06}), S_CURVE)


def test_the_stroke_is_as_dark_as_its_darkest_side():
    # a 2 px stroke is thinned onto its lighter top row
    dark = board(0.0, (20, slice(10, 190), 0.12), (21, slice(10, 190), 0.4))
    dark[45:49, 50:80] = 0.1
    assert not marks(dark, VIS, ALL)[0][46, 60]


def test_the_stroke_is_measured_on_the_board_not_round_it():
    # a dark frame just outside the read region must not raise the limits over a faint line
    img = picture(TURNED, [S_CURVE], 4, shade=FAINT)
    rim = np.int32(np.round(np.array(outline_px(HALF - 1.5, TURNED)) * 16))
    cv2.polylines(img, [rim], True, 60, 4, cv2.LINE_AA, 4)
    follows(run(img, TURNED), S_CURVE)


def test_a_blank_board_has_no_ink_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not np.any(marks(np.zeros(ALL.shape, np.float32), VIS, ALL))


def test_faint_wipe_ticks_in_a_row_are_not_skipped_but_one_faint_stroke_is():
    ticks, x = [], -100.0
    for k, (long, gap) in enumerate(zip((18, 21, 25, 19, 23), (10, 12, 14, 11, 13), strict=True)):
        ticks.append([(x, 60.0), (x + long, 60.0 + (3 if k % 2 else -3))])
        x += long + gap
    line = picture(VIS, [STRAIGHT], 4, shade=120)
    got = run(np.minimum(line, picture(VIS, ticks, 4, shade=FAINT)))
    follows(got, STRAIGHT)
    assert got.skipped == []
    one = [(-60.0, 60.0), (0.0, 60.0)]
    got = run(np.minimum(line, picture(VIS, [one], 4, shade=FAINT)))
    assert len(got.skipped) == 1 and along(got.skipped[0], one)[0].max() < 2


def test_faint_ink_outside_the_read_area_is_not_skipped():
    # on a turned board the read area's bounding box takes in board outside it
    h = HALF - 40.0
    img = np.minimum(
        picture(TURNED, [S_CURVE], 4, shade=120),
        picture(TURNED, [[(-50.0, h + 5), (50.0, h + 5)]], 4, shade=FAINT),
    )
    got = run(img, TURNED, edge=40.0)
    follows(got, S_CURVE)
    assert got.skipped == []
