"""The answer of one read of a drawn line, and the one several reads of the same board agree on."""

import math
from collections import Counter
from typing import NamedTuple

import numpy as np

DIFFER = "pictures do not agree"


class LinePath(NamedTuple):
    points: list[list[float]]
    reason: str
    skipped: list[list[list[float]]] = []  # strokes left out of the route, plate mm


def closes(points) -> int | None:
    """The first index whose point equals the last one, where the lap starts; None when open."""
    last = tuple(points[-1]) if len(points) else None
    return next((j for j in range(len(points) - 1) if tuple(points[j]) == last), None)


def _gaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each of a's points to the polyline b."""
    p, s, e = a[:, None], b[:-1], b[1:]
    seg = e - s
    t = np.clip(((p - s) * seg).sum(-1) / np.maximum((seg * seg).sum(-1), 1e-9), 0, 1)
    return np.linalg.norm(p - s - t[..., None] * seg, axis=-1).min(1)


def _inner(a: np.ndarray, keep: float, inside: bool = True) -> np.ndarray:
    """a's points more than keep from a free end (or not, inside False). A closed path has no
    free end where it closes, and its start loses at most half of the run into the lap."""
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
    k = closes(a.tolist())
    head = keep if k is None else min(keep, along[k] / 2) if k else -1.0
    end = along[-1] - keep if k is None else math.inf
    return a[((along > head) & (along < end)) == inside]


def _stray(a: np.ndarray, b: np.ndarray, tol: float) -> int:
    """Most points of a in a row that lie farther than tol from b."""
    run = most = 0
    for far in _gaps(a, b) > tol:
        run = run + 1 if far else 0
        most = max(most, run)
    return most


def _size(points) -> float:
    return float(np.linalg.norm(np.diff(np.array(points), axis=0), axis=1).sum())


def _vote(reads: list[LinePath], cfg: dict) -> LinePath | None:
    """The path most of the reads agree on, DIFFER when an early stop may have won, None when too
    few agree."""
    got = [r for r in reads if not r.reason]
    paths = [np.array(r.points) for r in got]
    shut = [closes(r.points) is not None for r in got]
    tol, keep = cfg["line_step_mm"] / 2, cfg["line_bridge_mm"]
    off = [[_gaps(_inner(a, keep), b) for b in paths] for a in paths]
    far = [[o.max(initial=0) for o in row] for row in off]
    gap = [[max(f, far[j][i]) for j, f in enumerate(row)] for i, row in enumerate(far)]
    votes = [
        sum(g <= tol and shut[i] == shut[j] for j, g in enumerate(row)) for i, row in enumerate(gap)
    ]
    if not (2 * len(paths) > len(reads) and 2 * max(votes) > len(paths)):
        return None
    top = [k for k, v in enumerate(votes) if v == max(votes)]
    # the middles are measured already, so the whole-path gap only needs the two end stretches
    rim = [_inner(p, keep, False) for p in paths]
    apart = {
        k: sum(
            max(gap[k][j], _gaps(rim[k], b).max(initial=0), _gaps(rim[j], paths[k]).max(initial=0))
            for j, b in enumerate(paths)
        )
        for k in top
    }
    best = min(top, key=apart.get)
    past = [_stray(b, paths[best], tol) * cfg["line_step_mm"] > keep for b in paths]
    on = sum(far[best][j] <= tol and past[j] for j in range(len(paths)))
    if on > 1 or (on and votes[best] < 3):
        return LinePath([], DIFFER)
    return LinePath(paths[best].tolist(), "", got[best].skipped)


def agree(reads: list[LinePath], cfg: dict) -> LinePath:
    """The path most of several reads of the same board agree on. Two agree when both close or
    neither does and every point of each, more than line_bridge_mm from a free end (where ends
    wander), lies within half a step of the other. More reads give a path than refuse and most of
    those agree: of the most agreed on, the one whose whole path, ends too, is closest to the
    rest. It is refused (DIFFER) when a read runs along it and then goes on for more than
    line_bridge_mm where it does not, or, once three reads or more agree on it, when two reads or
    more do: on a still board the same early stop comes back, so a majority is no proof.
    A read that left out at least line_similar of its route's length is not sure of it (the line
    may have broken in that picture): the sure reads vote first, all of them only when the sure
    ones give no answer. Too few agreeing gives the refusal most reads gave, or DIFFER."""
    unsure = [
        not r.reason and sum(map(_size, r.skipped)) >= cfg["line_similar"] * _size(r.points)
        for r in reads
    ]
    got = _vote(
        [LinePath([], "unsure") if u else r for r, u in zip(reads, unsure, strict=True)], cfg
    )
    if got is None and any(unsure):
        got = _vote(reads, cfg)
    if got is not None:
        return got
    said = Counter(r.reason for r in reads if r.reason).most_common(1)
    return LinePath([], said[0][0] if said and 2 * said[0][1] > len(reads) else DIFFER)
