import cv2
import numpy as np
import pytest

from src.io.settings import ROOT, load
from src.vision.balls import downsample, find_balls
from tools.replay_clip import run

pytestmark = pytest.mark.replay

CLIP = ROOT / "data/clips/lacrosse_ball_hand_rolled_1948.avi"
CFG = load()["vision"]


@pytest.fixture(scope="module")
def cap():
    if not CLIP.exists():
        pytest.skip(f"{CLIP.name} not on this machine")
    cap = cv2.VideoCapture(str(CLIP))
    yield cap
    cap.release()


def frame(cap, i: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read()
    assert ok
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def test_lacrosse_ball_at_the_edge_is_found_against_the_background(cap):
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    spaced = np.linspace(0, n - 1, CFG["bg_frames"]).astype(int)
    background = np.median([downsample(frame(cap, i), CFG) for i in spaced], axis=0)
    gray = frame(cap, 13384)
    assert find_balls(gray, CFG) == []
    found = find_balls(gray, CFG, background.astype(np.uint8))
    assert found and found[0].r_px == pytest.approx(79, abs=4)
    assert (found[0].x_px, found[0].y_px) == pytest.approx((779, 211), abs=4)


def test_lacrosse_ball_rolling_along_the_edge_is_tracked(cap):
    # the hand nudges the ball around 13386-13408 and takes it back from 13432 on
    ((radii, missed),) = run(str(CLIP), [(13310, 13430)], CFG).values()
    assert len(radii) / (len(radii) + len(missed)) >= 0.85
