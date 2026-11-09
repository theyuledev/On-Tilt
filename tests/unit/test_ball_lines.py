"""Whiteboard maze lines under and beside the ball (#148). Stroke widths 10 / 11 / 15 px are the
regression pins and stay literal: they hang on min_ball_r_px = 10, which the issue keeps."""

import cv2
import pytest

from src.io.settings import load
from src.vision.balls import find_balls
from src.vision.tracker import BallTracker
from tests.unit.test_ball_tracker import DT
from tests.unit.test_balls import CFG, ball, plate

pytestmark = pytest.mark.unit

LINE = 20
SHADE = 26
X, Y = 640, 400
# squash and small balls, test sizes only: the rig ball is rig.ball_d_mm
SQUASH_D_MM = 38.0
SMALL_D_MM = 13.0


def radius(d_mm):
    return round(d_mm * CFG["px_per_mm"] / 2)


BIG = [radius(load()["rig"]["ball_d_mm"]), radius(SQUASH_D_MM)]
SMALL = radius(SMALL_D_MM)


def hline(img, y, w):
    # filled rectangle, so a w px stroke is exactly rows y .. y+w-1 (cv2.line thickness isn't)
    cv2.rectangle(img, (140, y), (1139, y + w - 1), LINE, -1)
    return img


def vline(img, x, w):
    cv2.rectangle(img, (x, 40), (x + w - 1, 759), LINE, -1)
    return img


def crossing(w, r):
    return ball(hline(plate(), Y - w // 2, w), X, Y, r, shade=SHADE)


def tangent(w, r, below):
    return ball(hline(plate(), Y + r if below else Y - r - w + 1, w), X, Y, r, shade=SHADE)


def background(lines):
    """Tracker that has learned the lined board: bg_frames samples bg_period_s apart, else
    Background.update skips them. Returns it with the time the run can start from."""
    tr = BallTracker(CFG)
    period = CFG["bg_period_s"]
    for i in range(CFG["bg_frames"]):
        tr.update(lines(plate()), i * period)
    return tr, CFG["bg_frames"] * period


def near(found, r):
    return [c for c in found if abs(c.x_px - X) < r and abs(c.y_px - Y) < r]


@pytest.mark.parametrize("r", BIG)
@pytest.mark.parametrize("w", [4, 6, 8, 10])
def test_ball_on_a_stroke_up_to_10px_is_found_bare(w, r):
    (c,) = near(find_balls(crossing(w, r), CFG), r)
    assert c.x_px == pytest.approx(X, abs=2) and c.y_px == pytest.approx(Y, abs=2)


@pytest.mark.parametrize("r", BIG)
@pytest.mark.parametrize("w", [11, 15])
def test_ball_on_a_stroke_of_11px_or_more_is_lost_bare(w, r):
    assert near(find_balls(crossing(w, r), CFG), r) == []


@pytest.mark.parametrize("r", BIG)
@pytest.mark.parametrize("w", [4, 8, 10, 11])
def test_learned_background_erases_strokes_up_to_11px(w, r):
    tr, _ = background(lambda im: hline(im, Y - w // 2, w))
    (c,) = near(find_balls(crossing(w, r), CFG, tr.background.level), r)
    assert c.x_px == pytest.approx(X, abs=2) and c.y_px == pytest.approx(Y, abs=2)


@pytest.mark.parametrize("r", BIG)
def test_15px_stroke_loses_the_ball_even_against_the_background(r):
    w = 15
    tr, _ = background(lambda im: hline(im, Y - w // 2, w))
    assert near(find_balls(crossing(w, r), CFG, tr.background.level), r) == []


@pytest.mark.parametrize("below", [True, False])
@pytest.mark.parametrize("r", BIG)
def test_10px_stroke_tangent_to_a_big_ball_barely_pulls_it(r, below):
    (c,) = near(find_balls(tangent(10, r, below), CFG), r)
    assert abs(c.y_px - Y) < 4 and c.score > 0.78


@pytest.mark.parametrize("below", [True, False])
@pytest.mark.parametrize("w", [4, 8, 10])
def test_small_ball_tangent_to_any_stroke_is_too_ragged(w, below):
    img = tangent(w, SMALL, below)
    assert find_balls(img, CFG) == []
    # why: the stroke stuck to it drops its fill under min_fill, not just under the gate
    (c,) = near(find_balls(img, {**CFG, "min_fill": 0.0}), SMALL)
    assert c.score < 0.78


def track_over_verticals(w, r):
    """Ball rolls across three vertical strokes on a learned board. Returns per-frame
    (out, x, misses): misses is 0 only on a real hit, since out.valid stays true while coasting."""
    xs = (400, 640, 880)

    def lines(im):
        for x in xs:
            vline(im, x - w // 2, w)
        return im

    # stamps go on from the background's, so its sampling carries on through the run
    tr, t0 = background(lines)
    n = 360
    run = []
    for i in range(n):
        x = round(250 + (1030 - 250) * i / (n - 1))
        out = tr.update(ball(lines(plate()), x, Y, r, shade=SHADE), t0 + i * DT)
        run.append((out, x, tr.misses))
    return run


def lock_losses(run):
    was, losses = False, 0
    for out, _, _ in run:
        losses += was and not out.valid
        was = out.valid
    return losses


@pytest.mark.parametrize("r", BIG)
def test_tracked_ball_stays_locked_across_11px_strokes(r):
    run = track_over_verticals(11, r)
    for out, x, misses in run:
        # a hit every frame, not a coast through the stroke
        assert out.valid and misses == 0 and out.x_px == pytest.approx(x, abs=r / 2)


@pytest.mark.parametrize("r", BIG)
def test_tracked_ball_loses_lock_on_15px_strokes(r):
    assert lock_losses(track_over_verticals(15, r)) >= 1
