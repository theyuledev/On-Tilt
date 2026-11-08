import math

import pytest

from src.io.camera import ClipSource
from src.io.settings import ROOT, load
from src.vision.pipeline import Pipeline

pytestmark = pytest.mark.replay

CLIP = ROOT / "data" / "clips" / "squash_ball_hand_rolled_1944.avi"
START, END = 5130, 5370


@pytest.mark.skipif(not CLIP.exists(), reason="clip not on this machine")
def test_pipeline_tracks_the_hand_rolled_squash_ball_in_plate_mm():
    cfg = load()
    pipe = Pipeline(cfg)
    src = ClipSource(CLIP, start=START)
    obs = [pipe.step(*src.read()) for _ in range(END - START)]
    src.close()

    valid = [o.ball for o in obs if o.ball.valid]
    assert len(valid) >= 0.9 * (END - START)
    half = cfg["rig"]["plate_inner_mm"] / 2
    assert all(abs(s.x_mm) < half and abs(s.y_mm) < half for s in valid)
    span = cfg["vision"]["velocity_span_frames"]
    speeds = sorted(math.hypot(s.vx_mm_s, s.vy_mm_s) for s in valid[span:])
    assert 100 < speeds[len(speeds) // 2] < 1500 and speeds[-1] < 1500
