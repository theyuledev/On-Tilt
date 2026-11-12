import math

import cv2
import numpy as np
import pytest

from src.vision.px_to_mm import plate_px
from src.web.form import check
from tests.unit.line_pictures import (
    CFG,
    FAINT,
    S_CURVE,
    STRAIGHT,
    VIS,
    ball_px,
    follows,
    picture,
    run,
)
from tools.maze_from_frame import main

pytestmark = pytest.mark.unit

# a dark scribble: crossing strokes 8 mm apart in a 40 mm square
WEB = [[(-110.0 + k, -110.0), (-110.0 + k, -70.0)] for k in range(0, 41, 8)] + [
    [(-110.0, -110.0 + k), (-70.0, -110.0 + k)] for k in range(0, 41, 8)
]
# a smudge: darker than the board but nowhere as dark as a line, about 0.04 of it
SMUDGE = 197


def test_short_dark_marks_beside_the_line_are_not_lines():
    specks = [[(x, -60.0), (x + 12, -60.0)] for x in (-80.0, -30.0, 20.0)]
    follows(run(picture(VIS, [STRAIGHT, *specks])), STRAIGHT)


@pytest.mark.parametrize("touching", [False, True])
def test_a_darker_scribble_next_to_the_line_is_left_out(touching):
    line = [(-110.0 if touching else -60.0, -60.0), (80.0, 40.0)]
    lines = picture(VIS, [line], 3, shade=FAINT)
    follows(run(np.minimum(lines, picture(VIS, WEB, 3, shade=round(0.82 * 205)))), line[::-1])


def test_faint_smudges_with_no_dark_core_are_not_ink():
    marks = [[(-80.0, 80.0), (80.0, 80.0)], [(-60.0, -80.0), (60.0, -100.0)]]
    smudge = picture(VIS, marks, 8, shade=SMUDGE)
    follows(run(np.minimum(picture(VIS, [S_CURVE], 3, shade=FAINT), smudge)), S_CURVE)


def test_fainter_parts_joined_to_the_line_are_kept():
    faint = picture(VIS, [STRAIGHT], 3, shade=SMUDGE)
    follows(run(np.minimum(faint, picture(VIS, [STRAIGHT[:1] + [(-15.0, 5.0)]], 3))), STRAIGHT)


def test_a_shiny_ball_on_one_end_is_taken_out_whole():
    r_mm = 8.0
    img = picture(VIS, [STRAIGHT], ball=(STRAIGHT[0], r_mm))
    centre = np.int32(plate_px(*STRAIGHT[0], VIS))
    cv2.circle(img, centre, round(0.6 * r_mm * VIS["px_per_mm"]), 230, -1)
    follows(run(img, VIS, ball_px(VIS, STRAIGHT[0], r_mm)), STRAIGHT)


def test_small_dark_bits_round_the_ball_are_not_joined():
    x, y = STRAIGHT[0]
    spots = ((-4, 16.5), (-17, -3), (5, -16.5))
    bits = [[(x + dx, y + dy), (x + dx + 1, y + dy)] for dx, dy in spots]
    img = picture(VIS, [STRAIGHT, *bits], ball=(STRAIGHT[0], 8.0))
    follows(run(img, VIS, ball_px(VIS, STRAIGHT[0], 8.0)), STRAIGHT)


def test_a_thick_blob_that_is_not_the_ball_is_not_a_line():
    img = picture(VIS, [STRAIGHT], ball=((40.0, 60.0), 8.0))
    cv2.circle(img, np.int32(plate_px(-60.0, 60.0, VIS)), round(15 * VIS["px_per_mm"]), 30, -1)
    follows(run(img, VIS, ball_px(VIS, (40.0, 60.0), 8.0)), STRAIGHT)


def test_a_branch_off_the_line_is_left_out():
    bar = [(-80.0, 0.0), (80.0, 0.0)]
    follows(run(picture(VIS, [bar, [(0.0, 0.0), (0.0, 60.0)]])), bar)


CIRCLE = [[(50 * math.cos(a), 50 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 73)]]


@pytest.mark.parametrize(
    "lines, reason",
    [
        ([], "no line found"),
        ([[(-80.0, -30.0), (80.0, -30.0)], [(-80.0, 30.0), (80.0, 30.0)]], "more than one line"),
        (CIRCLE, "line branches or crosses itself"),
        ([[(0.0, 0.0), (25.0, 0.0)]], "line too short"),
    ],
)
def test_a_clear_reason_instead_of_a_wrong_path(lines, reason):
    got = run(picture(VIS, lines))
    assert got.reason == reason
    assert got.points == []


def test_the_tool_prints_the_line_off_a_saved_half_size_frame(tmp_path):
    frame = tmp_path / "board.png"
    cv2.imwrite(str(frame), cv2.resize(picture(VIS, [STRAIGHT], 8), (640, 400)))
    out = main([str(frame), "--line"])
    assert out.startswith("line ")
    assert "points = [[" in out
    assert [p.name for p in tmp_path.iterdir()] == ["board.png"]


def test_the_frame_and_the_map_it_is_given_are_left_as_they_were():
    img, cfg = picture(VIS, [STRAIGHT], ball=(STRAIGHT[0], 8.0)), {**VIS}
    before = img.copy()
    run(img, cfg, ball_px(VIS, STRAIGHT[0], 8.0))
    assert np.array_equal(img, before) and cfg == VIS


@pytest.mark.parametrize("key", ["line_stroke_mm", "line_min_mm", "line_step_mm"])
def test_the_settings_page_refuses_a_zero_size_or_step(key):
    with pytest.raises(ValueError, match="above 0"):
        check(CFG, {f"vision.{key}": 0})
