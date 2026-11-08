"""Progress of one run, advanced by the scoring step(progress, rec, maze, game, rig) -> Progress.

reached = gates passed in order, off_s = seconds off the path, done_t = t_s of the finish record,
out = ball left the plate, splits = t_s of each gate record, prev_t = t_s of the last record fed in.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from src.contracts import BallState


@dataclass(frozen=True, slots=True)
class Progress:
    reached: int = 0
    off_s: float = 0.0
    done_t: float | None = None
    out: bool = False
    splits: tuple[float, ...] = ()
    prev_t: float | None = None


def in_ring(ball: BallState, centre: tuple[float, float], r_mm: float) -> bool:
    return hypot(ball.x_mm - centre[0], ball.y_mm - centre[1]) <= r_mm
