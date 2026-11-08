import json
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path

from src.contracts import RunLogRecord
from src.games.progress import Progress, in_ring


def trace(records: Sequence[RunLogRecord]) -> list[list[float]]:
    # t runs from record 0 even if the ball was not seen yet
    if not records:
        return []
    t0 = records[0].t_s
    return [[r.t_s - t0, r.ball.x_mm, r.ball.y_mm] for r in records if r.ball.valid]


def at(points: list[list[float]], t: float) -> tuple[float, float] | None:
    if not points or t < points[0][0]:
        return None
    i = bisect_right(points, t, key=lambda p: p[0])
    if i == len(points):  # finished early, sits at the end
        return points[-1][1], points[-1][2]
    a, b = points[i - 1], points[i]
    f = (t - a[0]) / (b[0] - a[0])
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])


def save(records: Sequence[RunLogRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict()) + "\n")


def load(path: Path) -> list[RunLogRecord]:
    with path.open() as f:
        return [RunLogRecord.from_dict(json.loads(line)) for line in f if line.strip()]


def lap(records: Sequence[RunLogRecord], maze, game: dict, rig: dict, step) -> list[RunLogRecord]:
    # one lap = from the start ring exit to the finish record, the same cut the race makes
    i = _exit(records, maze, game)
    if i is None:
        return []
    p = Progress(prev_t=records[i].t_s)
    for j in range(i, len(records)):
        p = step(p, records[j], maze, game, rig)
        if p.done_t is not None:
            return list(records[i : j + 1])
    return []


def _exit(records: Sequence[RunLogRecord], maze, game: dict) -> int | None:
    r = game["checkpoint_r_mm"]
    valid = [i for i, rec in enumerate(records) if rec.ball.valid]
    inside = [in_ring(records[i].ball, maze.start, r) for i in valid]
    for k in range(1, len(valid)):
        if inside[k - 1] and not inside[k]:
            return valid[k]
    # a saved race already starts outside the ring
    if valid and not inside[0]:
        return valid[0]
    return None


def splits(records: Sequence[RunLogRecord], maze, game: dict, rig: dict, step) -> list[float]:
    if not records:
        return []
    p = Progress(prev_t=records[0].t_s)
    for rec in records:
        p = step(p, rec, maze, game, rig)
    return [s - records[0].t_s for s in p.splits]
