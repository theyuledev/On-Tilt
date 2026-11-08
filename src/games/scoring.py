from __future__ import annotations

from dataclasses import replace

from src.contracts import RunLogRecord
from src.games.progress import Progress, in_ring


def step(progress: Progress, rec: RunLogRecord, maze, game: dict, rig: dict) -> Progress:
    p = replace(progress, prev_t=rec.t_s)
    b = rec.ball
    if not b.valid or p.done_t is not None:
        return p
    half = rig["plate_inner_mm"] / 2
    if abs(b.x_mm) > half or abs(b.y_mm) > half:
        return replace(p, out=True)
    gate = maze.points[maze.checkpoints[p.reached]]
    if not in_ring(b, gate, game["checkpoint_r_mm"]):
        return p
    reached = p.reached + 1
    done = rec.t_s if reached == len(maze.checkpoints) else None
    return replace(p, reached=reached, splits=(*p.splits, rec.t_s), done_t=done)
