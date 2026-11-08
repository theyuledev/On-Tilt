import bisect
import math

import numpy as np
import pytest

from src.control.guard import Guard
from src.io.camera import ClipSource
from src.io.settings import ROOT, load
from src.vision.pipeline import Pipeline

pytestmark = pytest.mark.replay

CLIP = ROOT / "data/clips/lacrosse_ball_hand_rolled_1948.avi"


@pytest.fixture(scope="module")
def track():
    if not CLIP.exists():
        pytest.skip(f"{CLIP.name} not on this machine")
    cfg = load()
    src, pipe, out = ClipSource(str(CLIP), 13000), Pipeline(cfg), []
    for _ in range(460):
        f = src.read()
        out.append(pipe.step(f.image, f.t_s).ball)
    src.close()
    return out


def test_the_guard_sees_where_a_hand_rolled_ball_will_be(track):
    # the hand rolls it round the plate and along the edge up to ~13385, the plate stays level
    cfg = load()
    cfg["control"]["guard_on"] = True
    T = Guard(cfg).latency[0]
    ts = [b.t_s for b in track]
    moved, still = [], []
    for k, b in enumerate(track):
        j = bisect.bisect_left(ts, b.t_s + T)
        if j >= len(track) or not all(a.valid for a in track[k : j + 1]):
            continue
        a = track[j]
        moved.append(math.hypot(b.x_mm + b.vx_mm_s * T - a.x_mm, b.y_mm + b.vy_mm_s * T - a.y_mm))
        still.append(math.hypot(b.x_mm - a.x_mm, b.y_mm - a.y_mm))
    assert len(moved) > 200
    # seen 25.8 mm median, 43.5 p95 at T 0.27 s, against 42.4 mm for no velocity at all
    assert np.median(moved) < 30 and np.percentile(moved, 95) < 50
    assert np.median(moved) < np.median(still)
