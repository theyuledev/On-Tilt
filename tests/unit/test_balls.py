import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision.balls import downsample, find_balls

pytestmark = pytest.mark.unit

CFG = load()["vision"]
ANY_SHADE = {**CFG, "ball_dark": False}
H, W = 800, 1280


def plate(level=210, table=60):
    img = np.full((H, W), table, np.uint8)
    cv2.rectangle(img, (140, 40), (1140, 760), level, -1)
    return img


def ball(img, x, y, r, shade=25):
    cv2.circle(img, (x, y), r, shade, -1, cv2.LINE_AA)
    return img


def marker(img, x, y, r):
    """Thick lines under and just past the ball, plus a zigzag scribble beside it."""
    cv2.line(img, (x - 3 * r, y - r // 2), (x + 3 * r, y + r // 3), 20, 6, cv2.LINE_AA)
    cv2.line(img, (x + r + 10, y - 3 * r), (x + r + 10, y + 3 * r), 20, 8, cv2.LINE_AA)
    zig = np.array([(x - r - 120 + 15 * i, y + r + 20 + 30 * (i % 2)) for i in range(8)])
    cv2.polylines(img, [zig.astype(np.int32)], False, 20, 4, cv2.LINE_AA)
    return img


def only(found, x, y, r, dark=True):
    assert len(found) == 1, found
    c = found[0]
    assert c.x_px == pytest.approx(x, abs=2) and c.y_px == pytest.approx(y, abs=2)
    assert c.r_px == pytest.approx(r, abs=2.5)
    assert c.dark is dark


@pytest.mark.parametrize("r", [17, 57, 77])
def test_each_ball_size_is_found_among_marker_lines(r):
    img = marker(plate(), 640, 400, r)
    only(find_balls(ball(img, 640, 400, r), CFG), 640, 400, r)


def test_marker_lines_alone_are_not_a_ball():
    img = plate()
    for x, y, r in [(400, 300, 17), (700, 450, 57), (900, 250, 90)]:
        marker(img, x, y, r)
    assert find_balls(img, CFG) == []


def test_hand_is_not_a_ball():
    img = plate()
    cv2.ellipse(img, (640, 420), (40, 30), 0, 0, 360, 70, -1)
    for dx in (-30, -10, 10, 30):
        cv2.line(img, (640 + dx, 400), (640 + 2 * dx, 330), 70, 14)
    cv2.line(img, (640, 440), (640, 799), 70, 60)
    cv2.rectangle(img, (300, 300), (360, 340), 30, -1)
    assert find_balls(img, CFG) == []


def test_brightness_drift_and_uneven_light():
    img = ball(marker(plate(), 500, 300, 57), 500, 300, 57)
    ramp = np.linspace(0.35, 1.0, W)[None, :]
    only(find_balls((img * ramp).astype(np.uint8), CFG), 500, 300, 57)


@pytest.mark.parametrize("r, spot", [(17, 2), (57, 6)])
def test_specular_spot(r, spot):
    img = ball(plate(), 640, 400, r)
    cv2.circle(img, (640 - r // 3, 400 - r // 3), spot, 255, -1)
    only(find_balls(img, CFG), 640, 400, r)


def test_light_ball_on_dark_plate():
    img = ball(plate(level=70, table=25), 640, 400, 40, shade=170)
    only(find_balls(img, ANY_SHADE), 640, 400, 40, dark=False)


def test_glare_blob_on_whiteboard_is_no_ball_when_balls_are_dark():
    img = ball(plate(level=118), 640, 400, 40, shade=247)
    assert find_balls(img, {**CFG, "ball_dark": True}) == []
    only(find_balls(img, ANY_SHADE), 640, 400, 40, dark=False)


def test_small_glare_spot_is_not_a_ball():
    img = plate(level=120)
    cv2.circle(img, (640, 400), CFG["min_ball_r_px"] - 1, 255, -1)
    assert find_balls(img, CFG) == []


def test_dark_ball_wins_over_bright_one():
    img = ball(ball(plate(level=120), 400, 400, 40, shade=240), 800, 400, 40)
    only(find_balls(img, ANY_SHADE), 800, 400, 40)


def test_ball_at_the_plate_edge_is_only_found_against_the_background():
    # the table strip is narrower than the plate closing, so it gets the plate's level and
    # swallows a ball touching the edge; the real background keeps the two apart
    r = 45
    img = ball(plate(), 640, 40 + r, r)
    assert find_balls(img, CFG) == []
    only(find_balls(img, CFG, downsample(plate(), CFG)), 640, 40 + r, r)
