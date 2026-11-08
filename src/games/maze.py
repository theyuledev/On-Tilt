from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.io.settings import ROOT

MAZES = ROOT / "configs" / "mazes"


@dataclass(frozen=True, slots=True)
class Maze:
    points: tuple
    checkpoints: tuple
    start: tuple
    walls: tuple = ()


def _xy(p, what: str) -> tuple[float, float]:
    if len(p) != 2 or not all(isinstance(v, int | float) for v in p):
        raise ValueError(f"{what} must be [x, y], got {p!r}")
    return float(p[0]), float(p[1])


def load(name: str, root: Path = MAZES) -> Maze:
    path = root / f"{name}.toml"
    with path.open("rb") as f:
        raw = tomllib.load(f)
    points = tuple(_xy(p, f"{path.name} point") for p in raw["points"])
    checkpoints = tuple(raw["checkpoints"])
    if not checkpoints:
        raise ValueError(f"{path.name}: checkpoints is empty")
    for i in checkpoints:
        if not isinstance(i, int) or not 0 <= i < len(points):
            raise ValueError(f"{path.name}: checkpoint {i!r} is not an index into points")
    walls = tuple(tuple(_xy(p, f"{path.name} wall point") for p in w) for w in raw.get("walls", []))
    return Maze(points, checkpoints, _xy(raw["start"], f"{path.name} start"), walls)
