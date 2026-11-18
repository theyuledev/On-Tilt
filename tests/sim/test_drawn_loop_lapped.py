import math

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


# measured in 40 s: 4 and 3 laps, at most 5.2 and 9.4 mm off the line, 16 mm from the tail's middle
# once the ball reached the seam
@pytest.mark.parametrize("name", DRAWN)
def test_a_loop_read_off_a_drawn_picture_is_lapped_on_its_line(name):
    img = lamps(picture(VIS, DRAWN[name], 4), VIS, [(100.0, 100.0), (-110.0, -100.0)])
    path = agree([run(img)] * 5, VIS).points
    k = closes(path)
    cfg = rig.rig_cfg(path, rig.RIG_CONTROL)
    mode, track = make_mode("path", cfg), []
    r = rig.follow(path, None, FAR, 40.0, cfg=cfg, mode=mode, track=track)
    assert k is not None and mode.laps >= 2 and r["off_max_mm"] < OFF_MM
    # the tail is run once: after the seam the ball never goes back along it
    seam = next(i for i, p in enumerate(track) if math.dist(p, path[k]) < 10)
    mid = ((path[0][0] + path[k][0]) / 2, (path[0][1] + path[k][1]) / 2)
    assert k == 0 or min(math.dist(p, mid) for p in track[seam:]) > 10
