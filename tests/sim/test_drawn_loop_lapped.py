import math

import numpy as np
import path_rig as rig
import pytest

from src.modes import make_mode
from src.vision.line_vote import agree, closes
from tests.unit.line_pictures import VIS, lamps, picture, run

pytestmark = pytest.mark.sim

FAR = (1e6, 1e6)  # a loop has no end to finish at
OFF_MM = 12.0


def ring(r, n=90):
    return [
        (r * math.cos(t), r * math.sin(t)) for t in (2 * math.pi * i / (n - 1) for i in range(n))
    ]


TAIL = [(-60.0 - 59 * 0.5, 59 * 0.866), (-60.0, 0.0)]
DRAWN = {"circle": [ring(50.0)], "tail into loop": [TAIL, ring(60.0)]}


def voted(lines):
    clean = lamps(picture(VIS, lines, 4), VIS, [(100.0, 100.0), (-110.0, -100.0)]).astype(float)
    rng = np.random.default_rng(5)
    noisy = [np.uint8(np.clip(clean + rng.normal(0, 4, clean.shape), 0, 255)) for _ in range(5)]
    return agree([run(img) for img in noisy], VIS).points


# measured in 40 s: 4 and 3 laps, at most 5.3 and 9.4 mm off the line, the tail started 0.7 mm
# from its free end and the ball kept 16 mm from its middle once at the seam
@pytest.mark.parametrize("name", DRAWN)
def test_a_loop_read_off_five_noisy_pictures_is_lapped_on_its_line(name):
    path = voted(DRAWN[name])
    k = closes(path)
    cfg = rig.rig_cfg(path, rig.RIG_CONTROL)
    mode, track = make_mode("path", cfg), []
    r = rig.follow(path, None, FAR, 40.0, cfg=cfg, mode=mode, track=track)
    assert k is not None and mode.laps >= 2 and r["off_max_mm"] < OFF_MM
    if name == "circle":
        assert k == 0
        return
    # the tail is run once: from its free end, and after the seam never back along it
    assert k > 0 and math.dist(path[0], TAIL[0]) < 15
    seam = next(i for i, p in enumerate(track) if math.dist(p, path[k]) < 10)
    mid = ((path[0][0] + path[k][0]) / 2, (path[0][1] + path[k][1]) / 2)
    assert min(math.dist(p, mid) for p in track[seam:]) > 10
