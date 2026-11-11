"""Maze file from one overhead frame of the drawn whiteboard maze.

Finds the S and E letters, takes the corridor path between them and maps it to plate mm the
same way the tracker maps the ball. A frame smaller than the camera's is scaled up first.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.io.settings import ROOT, load
from src.vision.maze_path import checkpoints, length, maze_toml, polyline
from src.vision.maze_read import STROKES, read
from src.vision.px_to_mm import plate_mm


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frame", type=Path)
    p.add_argument("--name", default="drawn")
    p.add_argument("--out", type=Path, help="default configs/mazes/<name>.toml")
    p.add_argument("--start", type=float, nargs=2, help="S in frame px, skips the letter search")
    p.add_argument("--end", type=float, nargs=2, help="E in frame px")
    p.add_argument("--stroke-mm", type=float, default=STROKES["stroke_mm"])
    p.add_argument(
        "--board", type=float, default=STROKES["board"], help="board is brighter than this x top"
    )
    p.add_argument(
        "--dark", type=float, default=STROKES["dark"], help="stroke is darker than this x board"
    )
    p.add_argument("--margin-mm", type=float, default=2.0, help="keep the path off the strokes")
    p.add_argument("--letter-mm", type=float, nargs=2, default=list(STROKES["letter_mm"]))
    p.add_argument("--cap-mm", type=float, default=10.0, help="clearance worth chasing")
    p.add_argument("--tol-mm", type=float, default=3.0)
    p.add_argument("--step-mm", type=float, default=30.0)
    p.add_argument("--gates", type=int, default=4)
    return p


def main(argv=None) -> str:
    args = parser().parse_args(argv)
    cfg = load()
    vis = cfg["vision"]
    gray = cv2.imread(str(args.frame), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"can't read {args.frame}")
    scale = cfg["rig"]["camera_width"] / gray.shape[1]
    m = read(gray, vis["px_per_mm"] / scale, vars(args), args.start, args.end)
    print(f"board box px {m.board_box}  S px {_r(m.s_px)}  E px {_r(m.e_px)}")
    if m.path is None:
        raise SystemExit("no path from S to E, check the S / E px or pass --start / --end")
    pts = polyline(_mm(m.path, scale, vis), args.tol_mm, args.step_mm)
    gates = checkpoints(pts, args.gates, 2 * cfg["game"]["checkpoint_r_mm"])
    narrow = 2 * min(m.clear_px) * scale / vis["px_per_mm"] + 2 * args.margin_mm
    print(f"path {length(pts):.0f} mm in {len(pts)} points, gates {gates}, {len(m.lines)} walls")
    print(f"narrowest corridor on the path {narrow:.0f} mm")
    out = args.out or ROOT / "configs" / "mazes" / f"{args.name}.toml"
    if out.stem == "square":
        raise SystemExit("square is the stock maze, pick another --name")
    out.write_text(maze_toml(pts, gates, [_mm(w, scale, vis) for w in m.lines]))
    print(f"wrote {out}")
    return str(out)


def _mm(line, scale: float, vis: dict) -> np.ndarray:
    return np.array([plate_mm(x * scale, y * scale, vis) for x, y in line])


def _r(p) -> tuple[int, int]:
    return round(p[0]), round(p[1])


if __name__ == "__main__":
    main()
