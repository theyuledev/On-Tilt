import math

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision.cross import cross_mm, find_cross
from src.vision.px_to_mm import plate_mm

pytestmark = pytest.mark.unit

VIS = load()["vision"]
PX_MM = VIS["px_per_mm"]
INK = 40


def board() -> np.ndarray:
    img = np.full((800, 1280), 60, np.uint8)
    img[100:700, 340:940] = 200
    return img


def strokes(img, at, size_mm=16.0, turn_deg=45.0, cross_deg=90.0):
    # two strokes through `at`, a marker width wide; 45 is an X, 0 a plus
    r = size_mm * PX_MM / 2
    for deg in (turn_deg, turn_deg + cross_deg):
        dx, dy = r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))
        a, b = (round(at[0] - dx), round(at[1] - dy)), (round(at[0] + dx), round(at[1] + dy))
        cv2.line(img, a, b, INK, round(3 * PX_MM))
    return img


def clutter(img):
    cv2.circle(img, (500, 300), round(8 * PX_MM), 20, -1)  # the ball
    cv2.line(img, (400, 600), (880, 610), INK, 7)  # a wall
    cv2.rectangle(img, (700, 150), (860, 330), INK, 7)
    for ch, at in (("S", (380, 150)), ("E", (880, 660))):
        cv2.putText(img, ch, at, cv2.FONT_HERSHEY_SIMPLEX, 1.2, INK, 4)
    # a T, an L and a lone stroke of letter size
    cv2.line(img, (400, 460), (440, 460), INK, 7)
    cv2.line(img, (420, 460), (420, 500), INK, 7)
    cv2.line(img, (500, 520), (500, 560), INK, 7)
    cv2.line(img, (500, 560), (540, 560), INK, 7)
    cv2.line(img, (600, 180), (640, 200), INK, 7)
    return img


@pytest.mark.parametrize("turn", [45, 30, 20, 70, 0])
@pytest.mark.parametrize("size", [10, 13, 16])
def test_an_x_is_found_at_its_crossing_however_it_is_turned(turn, size):
    at = (640.0, 420.0)
    found = find_cross(strokes(clutter(board()), at, size, turn), PX_MM)
    assert found is not None and math.dist(found, at) < 2.5


def test_a_lopsided_x_is_found_where_its_strokes_cross():
    img = clutter(board())
    cv2.line(img, (630, 405), (662, 441), INK, 7)
    cv2.line(img, (625, 436), (655, 408), INK, 7)
    found = find_cross(img, PX_MM)
    assert found is not None and math.dist(found, (641, 420)) < 4


def test_a_ball_walls_letters_and_single_strokes_are_not_an_x():
    assert find_cross(board(), PX_MM) is None
    assert find_cross(clutter(board()), PX_MM) is None


def test_two_strokes_squashed_into_a_narrow_cross_are_not_an_x():
    assert find_cross(strokes(board(), (640, 420), 19, 45, 25), PX_MM) is None


@pytest.mark.parametrize("size", [5, 24, 60])
def test_an_x_far_from_letter_size_is_left_alone(size):
    assert find_cross(strokes(board(), (640, 420), size), PX_MM) is None


def test_two_xs_are_nobodys_target():
    img = strokes(strokes(board(), (500, 300)), (800, 500))
    assert find_cross(img, PX_MM) is None


def test_an_x_under_the_ball_is_not_found():
    img = strokes(board(), (640, 420))
    cv2.circle(img, (646, 424), round(8 * PX_MM), 20, -1)
    assert find_cross(img, PX_MM) is None


def test_the_x_comes_back_in_plate_mm_from_a_colour_frame_too():
    at = (640 + 60 * PX_MM, 400 - 45 * PX_MM)
    gray = strokes(board(), at)
    for image in (gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)):
        x, y = cross_mm(image, VIS)
        assert (x, y) == pytest.approx(plate_mm(*at, VIS), abs=1.0)
        assert (x, y) == pytest.approx((60, 45), abs=1.0)
    assert cross_mm(board(), VIS) is None
