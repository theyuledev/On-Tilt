import cv2
import pytest

from src.vision.tracker import BG_MIN_SAMPLES, BallTracker
from tests.unit.test_balls import ANY_SHADE, CFG, ball, plate

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
    tr = BallTracker(ANY_SHADE)
    n = lock(tr, [ball(plate(level=120), 500, 400, 40) for _ in range(3)])
    swapped = ball(plate(level=120), 500, 400, 40, shade=240)
    outs = [tr.update(swapped, (n + i) * DT) for i in range(CFG["lost_after_misses"])]
    assert not outs[-1].valid
    assert tr.update(swapped, 10.0).valid and tr.dark is False


def test_glare_blob_on_whiteboard_never_locks_when_balls_are_dark():
    img = ball(plate(level=118), 640, 400, 40, shade=247)
    tr = BallTracker({**CFG, "ball_dark": True})
    assert not any(tr.update(img, i * DT).valid for i in range(3))
    assert BallTracker(ANY_SHADE).update(img, 0.0).valid


def test_resting_ball_stays_out_of_the_background_and_found_for_a_minute():
    tr = BallTracker(CFG)
    img = ball(plate(), 500, 400, 45)
    period = CFG["bg_period_s"]
    for i in range(int(2 * 60 / period) + 1):
        out = tr.update(img, i * period / 2)
        assert out.valid and tr.misses == 0 and out.x_px == pytest.approx(500, abs=2)
    level = tr.background.level
    assert level is not None and len(tr.background.samples) == CFG["bg_frames"]
    # the plate under the ball, not the ball
    assert level[400 // 5, 500 // 5] == pytest.approx(210, abs=5)


def test_reset_background_starts_over_and_the_ball_is_still_tracked():
    tr = BallTracker(CFG)
    img = ball(plate(), 640, 40 + 45, 45)
    period = CFG["bg_period_s"]
    for i in range(BG_MIN_SAMPLES):
        tr.update(plate(), i * period)
    t = BG_MIN_SAMPLES * period
    assert tr.background.level is not None and tr.update(img, t).valid
    tr.reset_background()
    assert tr.background.level is None and tr.background.samples == []
    assert tr.update(ball(plate(), 640, 400, 45), t + period).valid


def test_frame_stall_does_not_fling_the_prediction():
    tr = BallTracker(CFG)
    n = lock(tr, [ball(plate(), 400 + 15 * i, 400, 45) for i in range(3)])
    x = 400 + 15 * (n - 1) + 30
    t = (n - 1) * DT + 0.07
    assert tr.predict(t)[0] == pytest.approx(x, abs=45)
    out = tr.update(ball(plate(), x, 400, 45), t)
    assert out.valid and out.x_px == pytest.approx(x, abs=2)


def test_two_stamps_a_millisecond_apart_do_not_fling_the_prediction():
    tr = BallTracker(CFG)
    assert tr.update(ball(plate(), 400, 400, 45), 0.0).valid
    assert tr.update(ball(plate(), 410, 400, 45), DT).valid
    assert tr.update(ball(plate(), 420, 400, 45), DT + 0.0009).valid
    assert tr.predict(2 * DT + 0.0009)[0] == pytest.approx(430, abs=45)


def test_blob_cut_by_the_search_box_is_skipped_and_the_ball_inside_taken():
    tr = BallTracker(CFG)
    n = lock(tr, [ball(plate(), 400 + 5 * i, 400, 45) for i in range(3)])
    img = ball(plate(), 415, 400, 45)
    # a hand coming in from the right: its edge sits inside the box, the rest outside
    cv2.rectangle(img, (480, 300), (900, 500), 25, -1)
    out = tr.update(img, n * DT)
    assert out.valid and out.x_px == pytest.approx(415, abs=2)
    # the ball itself half out of the box: found whole by the full frame pass instead
    out = tr.update(ball(plate(), 415 + 90, 400, 45), (n + 1) * DT)
    assert out.valid and out.x_px == pytest.approx(505, abs=2)
    assert out.r_px == pytest.approx(45, abs=2)


def test_ball_at_the_frame_edge_is_found_with_the_box_clamped():
    tr = BallTracker(CFG)
    n = lock(tr, [ball(plate(), 60 - 5 * i, 400, 45) for i in range(3)])
    out = tr.update(ball(plate(), 45, 400, 45), n * DT)
    assert out.valid and tr.misses == 0 and out.x_px == pytest.approx(45, abs=2)
