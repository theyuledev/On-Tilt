import math

import cv2
import numpy as np
import pytest

from src.vision.px_to_mm import plate_px
from src.vision.tracker import BallPx
from tests.unit.line_pictures import (
    BIG,
    CORNER,
    EDGE,
    FAINT,
    HALF,
    S_CURVE,
    SPIRAL,
    STRAIGHT,
    TURNED,
    VIS,
    along,
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
        (SPIRAL, 4, VIS),
    ],
)
def test_centre_of_one_stroke_comes_back_as_an_even_path(line, width, cfg):
    follows(run(picture(cfg, [line], width), cfg), line)


@pytest.mark.parametrize("line", [S_CURVE, SPIRAL])
def test_a_faint_thin_line_is_read(line):
    follows(run(picture(VIS, [line], 3, shade=FAINT)), line)


def test_a_line_close_to_the_board_edge_is_read():
    y = HALF - 15 / VIS["px_per_mm"]
    line = [(-80.0, y), (80.0, y)]
    follows(run(picture(VIS, [line], 3, shade=FAINT)), line)


def test_a_line_running_off_the_board_is_read_up_to_the_edge():
    got = run(picture(VIS, [[(0.0, 0.0), (HALF + 10, 0.0)]]))
    assert got.reason == ""
    assert max(x for x, _ in got.points) == pytest.approx(HALF - EDGE, abs=5)


def test_a_colour_frame_reads_the_same():
    gray = picture(VIS, [STRAIGHT])
    follows(run(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)), STRAIGHT)


def test_a_small_gap_in_a_thin_stroke_is_bridged():
    follows(run(picture(VIS, [STRAIGHT], 3, cut_x_mm=0.0)), STRAIGHT)


@pytest.mark.parametrize("r_mm", [8.0, BIG])
@pytest.mark.parametrize("cfg, width", [(VIS, 6), (TURNED, 10)])
def test_a_ball_on_one_end_neither_bends_nor_shortens_the_path(r_mm, cfg, width):
    img = picture(cfg, [STRAIGHT], width, ball=(STRAIGHT[0], r_mm))
    got = run(img, cfg, ball_px(cfg, STRAIGHT[0], r_mm))
    follows(got, STRAIGHT)
    assert min(math.dist(got.points[0], STRAIGHT[0]), math.dist(got.points[-1], STRAIGHT[0])) < 5


@pytest.mark.parametrize("r_mm", [8.0, BIG])
def test_a_ball_not_given_leaves_the_path_at_its_rim(r_mm):
    got = run(picture(VIS, [STRAIGHT], ball=(STRAIGHT[0], r_mm)))
    assert along(got.points, STRAIGHT)[0].max() < 2
    near = min(math.dist(got.points[0], STRAIGHT[0]), math.dist(got.points[-1], STRAIGHT[0]))
    assert near < r_mm + 5


@pytest.mark.parametrize("r_mm, given", [(8.0, True), (BIG, True), (8.0, False)])
def test_a_ball_on_the_middle_of_the_line_is_bridged(r_mm, given):
    mid = (-15.0, 5.0)
    img = picture(VIS, [STRAIGHT], ball=(mid, r_mm))
    follows(run(img, VIS, ball_px(VIS, mid, r_mm) if given else None), STRAIGHT)


@pytest.mark.parametrize("r_px", [8, 15, 30])
def test_glare_on_the_line_is_bridged(r_px):
    at = (10.0, 45 * math.sin(math.pi / 9.5))
    follows(run(picture(VIS, [S_CURVE], glare=(at, r_px))), S_CURVE)


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


def test_a_ball_given_picks_its_line_out_of_two():
    near, far = [(-80.0, -40.0), (60.0, -40.0)], [(-80.0, 60.0), (80.0, 60.0)]
    img = picture(VIS, [near, far], ball=(near[0], 8.0))
    follows(run(img, VIS, ball_px(VIS, near[0], 8.0)), near)


def test_a_ball_past_the_board_does_not_pull_the_end_out():
    line, at = [(0.0, 0.0), (HALF - 25, 0.0)], (HALF - 8, 0.0)
    got = run(picture(VIS, [line], ball=(at, 8.0)), VIS, ball_px(VIS, at, 8.0))
    assert max(x for x, _ in got.points) <= HALF - EDGE
    assert np.all(np.abs(np.array(got.points)[:, 1]) < 2)
