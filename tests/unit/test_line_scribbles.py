import math

import numpy as np
import pytest

from src.vision.line_path import BRANCH
from src.vision.maze_path import length
from tests.unit.line_pictures import (
    VIS,
    WINDING,
    ball_px,
    follows,
    lamps,
    picture,
    run,
    scribble,
    split,
)

pytestmark = pytest.mark.unit

BALL = ball_px(VIS, WINDING[0], 8)


def touched(cut):
    # a scribble touching the third row from below with a tail running off it, and a gap in the
    # row just past it, so the row's next piece is shorter than the tail
    tail = [(0.0, -51.5), (80.0, -59.5)]
    return scribble(-60.0, -21.5, 60.0, 30.0) + [tail] + split(WINDING, cut, -20.0)


@pytest.mark.parametrize("cut", [35.0, 20.0])
@pytest.mark.parametrize("ball", [BALL, None])
def test_a_scribble_touching_the_line_does_not_take_it_off_course(cut, ball):
    follows(run(picture(VIS, touched(cut), 4), ball=ball), WINDING)


@pytest.mark.parametrize("ball", [BALL, None])
def test_the_same_on_a_dimmer_board_with_lamp_spots_on_the_line(ball):
    img = lamps(
        np.uint8(picture(VIS, touched(35.0), 4) * 0.85), VIS, [(0, 100), (60, 40), (-60, -80)]
    )
    follows(run(img, ball=ball), WINDING)


@pytest.mark.parametrize("ball", [BALL, None])
def test_a_loop_hanging_off_a_scribble_on_the_line_is_not_taken(ball):
    loop = [
        (-30 + 10 * math.cos(t), 11.5 + 10 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 80)
    ]
    follows(
        run(picture(VIS, scribble(-60.0, 38.5, 60.0, 20.0) + [loop, WINDING], 4), ball=ball),
        WINDING,
    )


@pytest.mark.parametrize("ball", [BALL, None])
def test_a_scribble_near_the_line_end_does_not_turn_it_back_along_another_mark(ball):
    back = [(-30.0, -106.5), (110.0, -112.0)]
    marks = scribble(-70.0, -81.5, 40.0, 25.0) + [back, WINDING]
    follows(run(picture(VIS, marks, 4), ball=ball), WINDING)


@pytest.mark.parametrize("ball", [BALL, None])
def test_a_scribble_over_a_bend_or_over_one_line_ending_by_another_is_refused(ball):
    # the two look the same once the scribble is wiped, so neither is guessed
    bend = scribble(95.0, -25.0, 30.0, 50.0) + [WINDING]
    other = scribble(-115.0, -70.0, 30.0, 20.0) + [[(-100.0, -70.0), (40.0, -40.0)], WINDING]
    for marks in (bend, other):
        assert run(picture(VIS, marks, 4), ball=ball).reason != ""


@pytest.mark.parametrize("ball", [BALL, None])
def test_a_mark_off_the_line_before_a_scribble_on_its_bend_is_not_taken_for_it(ball):
    # the walk can't get across the scribble, so the mark is longer than what is left before it
    off = [(30.0, -20.0), (60.0, -14.0), (75.0, 0.0), (78.0, 25.0)]
    marks = scribble(95.0, -25.0, 30.0, 50.0) + [off, WINDING]
    assert run(picture(VIS, marks, 4), ball=ball).reason != ""


def test_a_gap_at_a_corner_with_a_bit_of_ink_in_it_is_refused_not_cut_short():
    marks = [
        [(-100.0, 0.0), (43.0, 0.0)],
        [(50.0, 2.0), (50.0, 8.0)],
        [(50.0, 15.0), (50.0, 120.0)],
    ]
    assert run(picture(VIS, marks, 4), ball=ball_px(VIS, (-100.0, 0.0), 8)).reason == BRANCH


@pytest.mark.parametrize("other", [[(-80.0, 40.0), (80.0, 40.0)], [(80.0, 40.0), (80.0, 140.0)]])
def test_another_line_ending_near_the_one_with_the_ball_is_left_out(other):
    line = [(-80.0, 0.0), (80.0, 0.0)]
    follows(run(picture(VIS, [line, other], 4), ball=ball_px(VIS, line[0], 8)), line)


@pytest.mark.parametrize("ball", [BALL, None])
def test_a_short_mark_just_past_the_line_end_is_no_reason_to_refuse(ball):
    line = [(-100.0, 0.0), (80.0, 0.0)]
    got = run(
        picture(VIS, [line, [(85.0, -7.5), (85.0, 7.5)]], 4), ball=ball and ball_px(VIS, line[0], 8)
    )
    assert got.reason == ""
    assert abs(length(got.points) / length(line) - 1) < 0.05


def test_a_corner_too_sharp_to_follow_is_refused_not_cut_short():
    v = [(-100.0, 0.0), (60.0, 0.0), (-80.0, 40.0)]
    assert run(picture(VIS, [v], 4), ball=ball_px(VIS, v[0], 8)).reason == BRANCH


def test_a_hook_too_sharp_to_follow_but_shorter_than_a_bridge_is_left_off():
    t = math.radians(191.6)
    v = [(-140.0, -100.0), (120.0, 60.0), (120 + 58 * math.cos(t), 60 + 58 * math.sin(t))]
    got = run(picture(VIS, [v], 4), ball=ball_px(VIS, v[0], 8))
    assert got.reason == ""
    assert abs(length(got.points) / length(v[:2]) - 1) < 0.03
