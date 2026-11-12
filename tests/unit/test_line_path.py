import math

import cv2
import pytest

from src.vision.px_to_mm import plate_px
from src.vision.tracker import BallPx
from tests.unit.line_pictures import (
    CORNER,
    EDGE,
    HALF,
    S_CURVE,
    STRAIGHT,
    TURNED,
    VIS,
    ball_px,
    follows,
    picture,
    run,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "line, width, cfg",
    [
        (STRAIGHT, 6, VIS),
        (S_CURVE, 6, VIS),
        (CORNER, 6, VIS),
        (S_CURVE, 14, VIS),
        (S_CURVE, 3, VIS),
        (S_CURVE, 6, TURNED),
        (CORNER, 10, TURNED),
    ],
)
def test_centre_of_one_stroke_comes_back_as_an_even_path(line, width, cfg):
    follows(run(picture(cfg, [line], width), cfg), line)


def test_a_colour_frame_reads_the_same():
    gray = picture(VIS, [STRAIGHT])
    follows(run(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)), STRAIGHT)


def test_a_small_gap_in_a_thin_stroke_is_bridged():
    follows(run(picture(VIS, [STRAIGHT], 3, cut_x_mm=0.0)), STRAIGHT)


@pytest.mark.parametrize("r_mm", [8.0, EDGE])
@pytest.mark.parametrize("given", [True, False])
@pytest.mark.parametrize("cfg, width", [(VIS, 6), (TURNED, 10)])
def test_a_ball_on_one_end_neither_bends_nor_shortens_the_path(r_mm, given, cfg, width):
    img = picture(cfg, [STRAIGHT], width, ball=(STRAIGHT[0], r_mm))
    got = run(img, cfg, ball_px(cfg, STRAIGHT[0], r_mm) if given else None)
    follows(got, STRAIGHT)
    assert min(math.dist(got.points[0], STRAIGHT[0]), math.dist(got.points[-1], STRAIGHT[0])) < 5


@pytest.mark.parametrize("r_mm", [8.0, EDGE])
@pytest.mark.parametrize("given", [True, False])
def test_a_ball_on_the_middle_of_the_line_is_bridged(r_mm, given):
    mid = (-15.0, 5.0)
    img = picture(VIS, [STRAIGHT], ball=(mid, r_mm))
    follows(run(img, VIS, ball_px(VIS, mid, r_mm) if given else None), STRAIGHT)


@pytest.mark.parametrize("r_px", [8, 15, 30])
def test_glare_on_the_line_is_bridged(r_px):
    follows(
        run(picture(VIS, [S_CURVE], glare=((10.0, 45 * math.sin(math.pi / 9.5)), r_px))), S_CURVE
    )


def test_glare_off_the_line_end_does_not_pull_it():
    img = picture(VIS, [STRAIGHT], glare=((STRAIGHT[1][0] + 12, STRAIGHT[1][1] + 4), 15))
    follows(run(img), STRAIGHT)


@pytest.mark.parametrize("given", [True, False])
def test_a_ball_off_the_line_is_ignored(given):
    line, at = [(-90.0, -40.0), (80.0, -40.0)], (40.0, 60.0)
    img = picture(VIS, [line], ball=(at, 8.0))
    follows(run(img, VIS, ball_px(VIS, at, 8.0) if given else None), line)


def test_a_lost_ball_is_not_used():
    near_end = (STRAIGHT[1][0] + 12, STRAIGHT[1][1] + 4)
    lost = BallPx(*plate_px(*near_end, VIS), 30.0, False, 0.0)
    follows(run(picture(VIS, [STRAIGHT]), VIS, lost), STRAIGHT)


@pytest.mark.parametrize("given", [True, False])
def test_a_ball_past_the_allowed_area_does_not_pull_the_end_out(given):
    line, at = [(0.0, 0.0), (HALF - EDGE - 20, 0.0)], (HALF - EDGE + 12, 0.0)
    img = picture(VIS, [line], ball=(at, 8.0))
    follows(run(img, VIS, ball_px(VIS, at, 8.0) if given else None), line)
