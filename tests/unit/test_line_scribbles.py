import math

import numpy as np
import pytest

from src.vision.line_path import BRANCH
from src.vision.maze_path import length
from tests.unit.line_pictures import (
    VIS,
    WINDING,
    along,
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


def test_a_scribble_over_a_sharp_bend_is_crossed_straight():
    got = run(picture(VIS, scribble(95.0, -25.0, 30.0, 50.0) + [WINDING], 4), ball=BALL)
    assert got.reason == ""
    assert abs(length(got.points) / length(WINDING) - 1) < 0.05
    pts = np.array(got.points)
    clear = (pts[:, 0] < 85) | (np.abs(pts[:, 1] + 50) > 35)
    assert along(pts[clear], WINDING)[0].max() < 2
    assert along(pts, WINDING)[0].max() < 30


def test_a_corner_too_sharp_to_follow_is_refused_not_cut_short():
    v = [(-100.0, 0.0), (60.0, 0.0), (-80.0, 40.0)]
    assert run(picture(VIS, [v], 4), ball=ball_px(VIS, v[0], 8)).reason == BRANCH


def test_a_hook_too_sharp_to_follow_but_shorter_than_a_bridge_is_left_off():
    v = [(-100.0, 0.0), (60.0, 0.0), (4.0, 20.0)]
    follows(run(picture(VIS, [v], 4), ball=ball_px(VIS, v[0], 8)), v[:2])
