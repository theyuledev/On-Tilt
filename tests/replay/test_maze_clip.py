import math
from itertools import pairwise

import cv2
import numpy as np
import pytest

from src.io.settings import ROOT, load
from src.vision.tracker import BallTracker

pytestmark = pytest.mark.replay

# 13 mm ball on a drawn maze, frames saved at half res as f_<unix time>.jpg. Seconds from the
# first frame: it rests at S until 15.15, and the board is empty from 37.42 to the end
CLIP = ROOT / "data/clips/maze_0340"
REST_UNTIL_S = 15.15
EMPTY_FROM_S = 37.42
CFG = load()["vision"]

# Each roll goes from the hand letting go (or the end of the rest) over the drawn walls and ends
# with the ball off the paper's right edge on the bare plate, before a hand catches it.
# Ball centre per unique frame in full-res px, from a separate threshold-and-link pass checked
# by eye; "-" is a frame where the paper's edge hides it
ROLLS = [
    (
        15.15,
        17.2,
        "661,514 660,514 666,512 671,510 675,509 680,508 690,505 699,502 709,501 "
        "720,497 732,493 746,489 757,483 777,477 793,472 808,465 831,461 847,451 866,447 892,438 "
        "908,431 925,425 914,414 - 928,403 922,393 942,393 948,383 961,378",
    ),
    (
        29.72,
        31.8,
        "451,470 456,467 461,464 467,461 475,457 484,453 494,449 505,444 519,441 "
        "531,436 546,431 562,426 580,420 599,414 620,407 643,400 666,393 685,384 708,378 737,368 "
        "763,360 790,353 813,344 845,336 866,328 891,320 883,321 - 907,314 911,312",
    ),
    (
        34.5,
        36.45,
        "469,465 472,467 475,471 481,471 486,472 494,471 504,470 512,471 528,469 "
        "538,468 551,466 569,463 585,459 600,455 622,449 641,443 660,437 681,430 704,424 729,414 "
        "753,407 777,398 805,389 829,381 858,370 886,362 903,352 892,348",
    ),
]
# 75 of the 85 labelled frames lock today; the misses are the ball off the paper, too faint on
# the bare plate, and the first frame after the hand lets go
ROLL_LOCK_MIN = 0.86
# labels top out at 252 mm/s
ROLL_MAX_MM_S = 350


@pytest.fixture(scope="module")
def tracked():
    if not CLIP.exists():
        pytest.skip(f"{CLIP.name} not on this machine")
    paths = sorted(CLIP.glob("f_*.jpg"), key=lambda p: float(p.stem[2:]))
    t0 = float(paths[0].stem[2:])
    tr = BallTracker(CFG)
    out, prev = [], None
    for p in paths:
        img = cv2.imread(str(p))
        # back to back copies are one camera frame saved twice
        if prev is not None and np.array_equal(img, prev):
            continue
        prev = img
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2)
        t = float(p.stem[2:])
        b = tr.update(gray, t)
        out.append((t - t0, b, b.valid and tr.misses == 0))
    return out


def test_empty_maze_never_shows_a_ball(tracked):
    empty = [b for t, b, _ in tracked if t >= EMPTY_FROM_S]
    assert len(empty) > 300
    assert not [b for b in empty if b.valid]


def test_small_ball_resting_at_the_start_is_seen_every_frame(tracked):
    # seen, not coasted: a coast at rest lands on the ball too and would hide the dips
    rest = [(b, seen) for t, b, seen in tracked if t < REST_UNTIL_S]
    x, y = np.median([(b.x_px, b.y_px) for b, seen in rest if seen], axis=0)
    tol = 4 * CFG["px_per_mm"]
    near = [b for b, seen in rest if seen and math.hypot(b.x_px - x, b.y_px - y) <= tol]
    assert len(near) / len(rest) >= 0.98


def rolls(tracked):
    # per roll: (t, ball, seen, mm off the label or None)
    out = []
    for t0, t1, labels in ROLLS:
        frames = [f for f in tracked if t0 <= f[0] <= t1]
        roll = []
        for (t, b, seen), m in zip(frames, labels.split(), strict=True):
            err = None
            if m != "-":
                x, y = map(float, m.split(","))
                err = math.hypot(b.x_px - x, b.y_px - y) / CFG["px_per_mm"]
            roll.append((t, b, seen, err))
        out.append(roll)
    return out


def test_rolling_ball_stays_locked(tracked):
    labelled = [(b, e) for roll in rolls(tracked) for _, b, _, e in roll if e is not None]
    locked = [b for b, e in labelled if b.valid and e <= 4]
    assert len(locked) / len(labelled) >= ROLL_LOCK_MIN


def test_rolling_ball_never_jumps(tracked):
    for roll in rolls(tracked):
        valid = [(t, b) for t, b, _, _ in roll if b.valid]
        for (t0, a), (t1, b) in pairwise(valid):
            step = math.hypot(b.x_px - a.x_px, b.y_px - a.y_px) / CFG["px_per_mm"]
            assert step <= ROLL_MAX_MM_S * (t1 - t0)


def test_ball_lost_off_the_paper_goes_invalid_not_somewhere_wrong(tracked):
    frames = [f for roll in rolls(tracked) for f in roll]
    off = [seen for _, b, seen, e in frames if b.valid and e is not None and e > 10]
    # a detection never lands off the ball. At 16 fps the misses before it gives up span 0.3 s,
    # so one held coast at the edge drifts past
    assert not any(off)
    assert len(off) <= 1
