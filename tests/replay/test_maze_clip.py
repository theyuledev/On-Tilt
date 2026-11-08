import math

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
