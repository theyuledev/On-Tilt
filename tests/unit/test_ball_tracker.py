import cv2
import pytest

from src.vision.tracker import BallTracker
from tests.unit.test_balls import CFG, ball, plate

pytestmark = pytest.mark.unit

DT = 1 / 120


def lock(tr, frames):
    for i, img in enumerate(frames):
        assert tr.update(img, i * DT).valid
    return len(frames)


def test_follows_a_moving_ball_over_a_line_and_keeps_its_radius():
    tr = BallTracker(CFG)
    for i in range(50):
        x, y = 300 + 12 * i, 380 + i
        img = plate()
        cv2.line(img, (640, 40), (640, 760), 20, 8, cv2.LINE_AA)
        if i > 2:
            ball(img, 700, 520, 17)
        out = tr.update(cv2.cvtColor(ball(img, x, y, 45), cv2.COLOR_GRAY2BGR), i * DT)
        # touching the line pulls the centre a little for a few frames
        assert out.valid
        assert out.x_px == pytest.approx(x, abs=3) and out.y_px == pytest.approx(y, abs=3)
        assert out.r_px == pytest.approx(45, abs=2)


def test_finds_the_ball_again_after_a_jump():
    tr = BallTracker(CFG)
    n = lock(tr, [ball(plate(), 400 + 5 * i, 400, 45) for i in range(3)])
    out = tr.update(ball(plate(), 900, 250, 45), n * DT)
    assert out.valid and out.x_px == pytest.approx(900, abs=2)


def test_coasts_then_goes_lost_then_picks_up_a_new_ball():
    tr = BallTracker(CFG)
    n = lock(tr, [ball(plate(), 400 + 5 * i, 400, 45) for i in range(3)])
    misses = [tr.update(plate(), (n + i) * DT) for i in range(CFG["lost_after_misses"])]
    assert misses[0].valid and misses[0].x_px == pytest.approx(415, abs=2)
    assert all(o.valid for o in misses[:-1]) and not misses[-1].valid
    out = tr.update(ball(plate(), 900, 300, 17), 10.0)
    assert out.valid and out.x_px == pytest.approx(900, abs=2)
    assert out.r_px == pytest.approx(17, abs=2)


def test_locked_on_a_dark_ball_never_takes_a_bright_one():
    tr = BallTracker(CFG)
    n = lock(tr, [ball(plate(level=120), 500, 400, 40) for _ in range(3)])
    swapped = ball(plate(level=120), 500, 400, 40, shade=240)
    outs = [tr.update(swapped, (n + i) * DT) for i in range(CFG["lost_after_misses"])]
    assert not outs[-1].valid
    assert tr.update(swapped, 10.0).valid and tr.dark is False
