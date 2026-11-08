import json
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path

from src.contracts import RunLogRecord


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
