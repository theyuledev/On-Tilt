"""The answer of one read of a drawn line, and the one several reads of the same board agree on."""

from collections import Counter
from typing import NamedTuple

import numpy as np

DIFFER = "pictures do not agree"


class LinePath(NamedTuple):
    points: list[list[float]]
    reason: str


def _gaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of a's points to the polyline b."""
    p, s, e = a[:, None], b[:-1], b[1:]
    seg = e - s
    t = np.clip(((p - s) * seg).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(p - s - t[..., None] * seg, axis=-1).min(1)


def _inner(a: np.ndarray, keep: float, inside: bool = True) -> np.ndarray:
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
    return a[((along > keep) & (along < along[-1] - keep)) == inside]


def _stray(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    """Most points of a in a row that lie farther than tol from b."""
    run = most = 0
    for far in _gaps(a, b) > tol:
        run = run + 1 if far else 0
        most = max(most, run)
    return most


def agree(reads: list[LinePath], cfg: dict) -> LinePath:
    """The path most of several reads of the same board agree on. Two agree when every point of
    each, more than line_bridge_mm from its ends (where the ends wander), lies within half a step
    of the other. More reads give a path than refuse and most of those agree: of the most agreed
    on, the one whose whole path, ends too, is closest to the rest. It is refused (DIFFER) when
    a read runs along it and then goes on for more than line_bridge_mm where it does not, or,
    once three reads or more agree on it, when two reads or more do: on a still board the same
    early stop comes back, so a majority is no proof. Too few agreeing gives the refusal most
    reads gave, or DIFFER."""
    paths = [np.array(r.points) for r in reads if not r.reason]
    tol, keep = cfg["line_step_mm"] / 2, cfg["line_bridge_mm"]
    off = [[_gaps(_inner(a, keep), b) for b in paths] for a in paths]
    far = [[o.max(initial=0) for o in row] for row in off]
    gap = [[max(f, far[j][i]) for j, f in enumerate(row)] for i, row in enumerate(far)]
    votes = [sum(g <= tol for g in row) for row in gap]
    if 2 * len(paths) > len(reads) and 2 * max(votes) > len(paths):
        top = [k for k, v in enumerate(votes) if v == max(votes)]
        # the middles are measured already, so the whole-path gap only needs the two end stretches
        rim = [_inner(p, keep, False) for p in paths]
        apart = {
            k: sum(
                max(gap[k][j], _gaps(rim[k], b).max(), _gaps(rim[j], paths[k]).max())
                for j, b in enumerate(paths)
            )
            for k in top
        }
        best = min(top, key=apart.get)
        past = [_stray(b, paths[best], tol) * cfg["line_step_mm"] > keep for b in paths]
        on = sum(far[best][j] <= tol and past[j] for j in range(len(paths)))
        if on > 1 or (on and votes[best] < 3):
            return LinePath([], DIFFER)
        return LinePath(paths[best].tolist(), "")
    said = Counter(r.reason for r in reads if r.reason).most_common(1)
    return LinePath([], said[0][0] if said and 2 * said[0][1] > len(reads) else DIFFER)
