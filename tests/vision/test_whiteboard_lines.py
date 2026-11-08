"""Marker strokes on a whiteboard (210) under a ball (26), ink 20, ball radii at 2.37 px/mm.
Strokes are filled rectangles so the width is exact: a thick cv2.line rounds it."""

import cv2
import pytest

from src.vision.balls import find_balls
from src.vision.tracker import BallTracker
from tests.unit.test_balls import CFG, plate

pytestmark = pytest.mark.unit

# 61.8 mm lacrosse, 38 mm squash; the 13 mm ball fails tangent to any line, it stays off the board
RADII = [73, 45]
X, Y = 640, 400


def stroke(img, x0, y0, x1, y1):
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), 20, -1)
    return img


def ball(img, x, y, r):
    cv2.circle(img, (round(x), round(y)), r, 26, -1, cv2.LINE_AA)
    return img


def near(c, x, y, tol=4):
    return abs(c.x_px - x) < tol and abs(c.y_px - y) < tol


@pytest.mark.parametrize("r", RADII)
@pytest.mark.parametrize("w", [2, 6, 10])
def test_stroke_crossing_the_ball_leaves_it_clean(r, w):
    for x0 in range(X - r - w + 1, X + r, r // 4):
        found = find_balls(ball(stroke(plate(), x0, 40, x0 + w, 760), X, Y, r), CFG)
        assert len(found) == 1 and near(found[0], X, Y), (x0, found)
        assert found[0].r_px == pytest.approx(r, abs=4)


@pytest.mark.parametrize("r", RADII)
@pytest.mark.parametrize("w", [2, 6, 10])
def test_stroke_tangent_to_the_ball_barely_pulls_it(r, w):
    found = find_balls(ball(stroke(plate(), 140, Y + r, 1140, Y + r + w), X, Y, r), CFG)
    assert len(found) == 1, found
    assert abs(found[0].y_px - Y) < 4 and found[0].score > 0.78


@pytest.mark.parametrize("r", RADII)
def test_tracked_ball_holds_lock_over_three_11px_strokes(r):
    board = plate()
    for x in (400, 640, 880):
        stroke(board, x, 40, x + 11, 760)
    tr = BallTracker(CFG)
    period = CFG["bg_period_s"]
    for i in range(30):
        tr.update(board, i * period)
    for i in range(360):
        x, y = 250 + 780 * i / 359, 380 + 40 * i / 359
        out = tr.update(ball(board.copy(), x, y, r), 30 * period + i / 120)
        # a detection every frame, not a coasted one
        assert out.valid and tr.misses == 0, i
        assert near(out, round(x), round(y), 6), (i, out)
