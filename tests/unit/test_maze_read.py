import math
import tomllib

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision import calib
from src.vision.maze_path import checkpoints
from src.vision.maze_read import read
from src.vision.px_to_mm import plate_mm
from tests.unit.test_px_to_mm import LENS
from tools.maze_from_frame import main, parser

pytestmark = pytest.mark.unit

# half size frame, 1.185 px per plate mm at the checked in px_per_mm
PX_MM = load()["vision"]["px_per_mm"] / 2
S, E = (200, 85), (415, 315)
# block A hangs from the top, block B stands on the bottom: S goes down, under A, up, over B, down
BLOCKS = [((230, 40), (290, 290)), ((320, 110), (360, 360))]


def drawn_maze() -> tuple[np.ndarray, np.ndarray]:
    img = np.full((400, 640), 60, np.uint8)
    img[50:350, 170:470] = 120
    walls = np.zeros_like(img)
    for a, b in BLOCKS:
        cv2.rectangle(img, a, b, 30, 3)
        cv2.rectangle(walls, a, b, 1, 3)
    for ch, (x, y) in (("S", S), ("E", E)):
        cv2.putText(img, ch, (x - 6, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 30, 2)
    cv2.line(img, (292, 170), (318, 150), 250, 6)
    # a ball sized blob filling the middle corridor is not a wall
    cv2.circle(img, (305, 230), 11, 20, -1)
    return img, walls.astype(bool) & (img < 120)


def test_path_runs_s_to_e_through_the_corridors_without_crossing_a_stroke():
    img, walls = drawn_maze()
    m = read(img, PX_MM, vars(parser().parse_args(["x.png"])))
    assert math.dist(m.s_px, S) < 5 and math.dist(m.e_px, E) < 5
    assert math.dist(m.path[0], S) < 10 * PX_MM and math.dist(m.path[-1], E) < 10 * PX_MM
    assert not any(walls[y, x] for x, y in m.path)
    assert all(170 <= x < 470 and 50 <= y < 350 for x, y in m.path)
    under_a = [y for x, y in m.path if 230 <= x <= 290]
    over_b = [y for x, y in m.path if 320 <= x <= 360]
    assert min(under_a) > 290 and max(over_b) < 110


def test_start_and_end_px_override_the_letters():
    img, _ = drawn_maze()
    m = read(img, PX_MM, vars(parser().parse_args(["x.png"])), (300, 300), E)
    assert m.s_px == (300, 300) and math.dist(m.path[0], (300, 300)) < 3


def test_written_maze_has_the_game_keys_and_spaced_gates(tmp_path):
    img, _ = drawn_maze()
    frame, out = tmp_path / "f.png", tmp_path / "drawn.toml"
    cv2.imwrite(str(frame), img)
    main([str(frame), "--out", str(out)])
    with out.open("rb") as f:
        maze = tomllib.load(f)
    assert set(maze) == {"points", "checkpoints", "start", "walls"}
    pts, gates = maze["points"], maze["checkpoints"]
    assert maze["start"] == pts[0] and gates[-1] == len(pts) - 1 and len(gates) >= 2
    cfg = load()
    gap = 2 * cfg["game"]["checkpoint_r_mm"]
    assert all(b - a >= 2 for a, b in zip(gates[:-1], gates[1:], strict=True))
    assert all(math.dist(pts[a], pts[b]) >= gap for a in gates for b in gates if a < b)
    e_mm = plate_mm(E[0] * 2, E[1] * 2, cfg["vision"])
    assert math.dist(pts[-1], e_mm) < 10 and len(maze["walls"]) == len(BLOCKS)


def test_square_is_not_overwritten(tmp_path):
    img, _ = drawn_maze()
    frame = tmp_path / "f.png"
    cv2.imwrite(str(frame), img)
    with pytest.raises(SystemExit):
        main([str(frame), "--out", str(tmp_path / "square.toml")])


def test_gates_keep_their_distance_on_a_line_that_doubles_back():
    pts = [(x, 0.0) for x in range(0, 200, 20)] + [(x, 10.0) for x in range(200, 80, -20)]
    gates = checkpoints(pts, 6, 60)
    assert gates[-1] == len(pts) - 1
    near = [0, *gates]
    assert all(math.dist(pts[a], pts[b]) >= 60 for a in near for b in near if a < b)
    assert all(b - a >= 2 for a, b in zip(gates[:-1], gates[1:], strict=True))


def test_a_lens_maps_the_walls_like_the_tracker_maps_the_ball(tmp_path, monkeypatch):
    img, _ = drawn_maze()
    frame = tmp_path / "f.png"
    cv2.imwrite(str(frame), img)
    lens = tmp_path / "calib.local.json"
    monkeypatch.setattr(calib, "CALIB_PATH", lens)
    main([str(frame), "--out", str(tmp_path / "flat.toml")])
    calib.save_calib(LENS, lens)
    main([str(frame), "--out", str(tmp_path / "bent.toml")])
    flat, bent = (tomllib.loads((tmp_path / f"{n}.toml").read_text()) for n in ("flat", "bent"))
    vis = load()["vision"]
    (cx, cy), s = vis["plate_centre_px"], vis["px_per_mm"]
    for fw, bw in zip(flat["walls"], bent["walls"], strict=True):
        px = [(cx + x * s, cy - y * s) for x, y in fw]
        assert np.allclose([plate_mm(u, v, vis, LENS) for u, v in px], bw, atol=0.5)
    assert not np.allclose(flat["walls"], bent["walls"], atol=1)
