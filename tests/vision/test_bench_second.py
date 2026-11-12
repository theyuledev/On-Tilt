import cv2
import numpy as np
import pytest

from src.vision.bench.second import disc

pytestmark = pytest.mark.unit

R = 19


def plate() -> np.ndarray:
    """A light board with a little grain on a dark floor."""
    grain = np.random.default_rng(1).integers(-4, 5, (300, 400))
    out = np.full((300, 400), 60, np.int16)
    out[40:260, 60:340] = 200
    return (out + grain).astype(np.uint8)


def ball(img: np.ndarray, x: int, y: int, r: int = R) -> np.ndarray:
    cv2.circle(img, (x, y), r, 30, -1)
    return img


def test_disc_match_finds_a_drawn_ball_on_a_wide_stroke():
    img = plate()
    cv2.line(img, (60, 150), (340, 150), 30, 15)
    x, y, r = disc(ball(img, 220, 150), [R])
    assert (x, y) == pytest.approx((220, 150), abs=3) and r == R


def test_disc_match_finds_a_drawn_ball_at_the_board_edge():
    x, y, _ = disc(ball(plate(), 60 + R, 150), [R])
    assert (x, y) == pytest.approx((60 + R, 150), abs=3)


def test_the_best_of_several_sizes_is_the_drawn_balls():
    # 90 is too big to slide over this picture at all and is passed over
    x, y, r = disc(ball(plate(), 200, 150, 30), [12, 30, 45, 90])
    assert (x, y, r) == pytest.approx((200, 150, 30), abs=3)


def test_it_answers_on_an_empty_plate_too():
    x, y, r = disc(plate(), [R])
    assert 0 <= x < 400 and 0 <= y < 300 and r == R


def test_a_picture_too_small_to_slide_the_disc_over_gets_no_answer():
    assert disc(plate()[:60, :60], [R]) is None
