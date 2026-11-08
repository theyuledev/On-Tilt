import math

import cv2
import pytest

from src.io.settings import ROOT, load
from src.vision.maze_read import read
from tools.maze_from_frame import parser

pytestmark = pytest.mark.replay

SNAP = ROOT / "data" / "snaps" / "maze_whiteboard_0327.jpg"
# letters as they sit in the half size snap, read off by eye
S, E = (331, 271), (433, 226)


@pytest.mark.skipif(not SNAP.exists(), reason="snap not on this machine")
def test_drawn_maze_path_runs_from_s_to_e_without_crossing_a_stroke():
    gray = cv2.imread(str(SNAP), cv2.IMREAD_GRAYSCALE)
    px_mm = load()["vision"]["px_per_mm"] * gray.shape[1] / load()["rig"]["camera_width"]
    m = read(gray, px_mm, vars(parser().parse_args([str(SNAP)])))
    assert math.dist(m.s_px, S) < 5 and math.dist(m.e_px, E) < 5
    assert m.path is not None
    assert math.dist(m.path[0], S) < 10 * px_mm and math.dist(m.path[-1], E) < 10 * px_mm
    assert not any(m.walls[y, x] for x, y in m.path)
    # the stroke ends reach the sheet's edge, so the way round goes up through the maze
    assert min(y for _, y in m.path) < 100
