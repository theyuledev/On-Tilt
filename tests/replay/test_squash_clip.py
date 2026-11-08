import math

import pytest

from src.io.camera import ClipSource
from src.io.settings import ROOT, load
from src.vision.px_to_mm import SPAN, ball_state
from src.vision.tracker import BallTracker

pytestmark = pytest.mark.replay

CLIP = ROOT / "data" / "clips" / "squash_ball_hand_rolled_1944.avi"
START, END = 5130, 5370


@pytest.mark.skipif(not CLIP.exists(), reason="clip not on this machine")
def test_hand_rolled_squash_ball_tracks_in_plate_mm():
    cfg = load()
    tracker = BallTracker(cfg["vision"])
    src = ClipSource(CLIP, start=START)
    states = []
    for _ in range(END - START):
        f = src.read()
        states.append(ball_state(tracker.update(f.image, f.t_s), states[-SPAN:], cfg["vision"]))
    src.close()

    valid = [s for s in states if s.valid]
    assert len(valid) >= 0.9 * (END - START)
    half = cfg["rig"]["plate_inner_mm"] / 2
    assert all(abs(s.x_mm) < half and abs(s.y_mm) < half for s in valid)
    speeds = sorted(math.hypot(s.vx_mm_s, s.vy_mm_s) for s in valid[SPAN:])
    assert 100 < speeds[len(speeds) // 2] < 1500 and speeds[-1] < 1500
