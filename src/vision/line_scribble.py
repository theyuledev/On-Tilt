"""Scribbles on a drawn line: wiped where forks crowd together, and the ways on through them."""

import cv2
import numpy as np

from src.vision.line_graph import chains


def pieces(marks: np.ndarray, stroke: int, web: float = 0.0):
    """Chains of the marks' skeleton, with specks and short spurs dropped. Where three forks or
    more sit within web px of each other (a scribble, not a line) the marks are wiped and read
    again; the wiped area comes back too (None when nothing was wiped)."""
    found = chains(cv2.ximgproc.thinning(np.uint8(marks) * 255) > 0, 3 * stroke)
    degree: dict[int, int] = {}
    at = {}
    for c in found:
        for e, node in enumerate(c.nodes):
            degree[node] = degree.get(node, 0) + 1
            at[node] = c.px[-e]
    forks = np.array([at[n] for n, d in degree.items() if d >= 3]).reshape(-1, 2)
    close = np.linalg.norm(forks[:, None] - forks[None], axis=-1) <= web
    wipe = forks[close.sum(1) >= 3] if web else forks[:0]
    if len(wipe):
        wiped = np.zeros_like(marks)
        for x, y in wipe:
            cv2.circle(wiped, (round(x), round(y)), round(web / 2), 1, -1)
        return pieces(marks & (1 - wiped), stroke)[0], wiped
    return [c for c in found if c.length >= stroke or min(degree[n] for n in c.nodes) > 1], None


def zones(found: list, wiped, reach: int) -> dict:
    """Which wiped scribble, 1, 2, ..., each chain end lies within reach px of; 0 for none."""
    if wiped is None:
        return {}
    _, lab = cv2.connectedComponents(cv2.dilate(wiped, np.ones((2 * reach + 1,) * 2, np.uint8)))
    h, w = lab.shape
    out = {}
    for i, c in enumerate(found):
        for e in (0, 1):
            x, y = np.clip(np.round(c.px[-e]).astype(int), 0, [w - 1, h - 1])
            out[i, e] = int(lab[y, x])
    return out


def through(zone: dict, joined: dict) -> dict:
    """Join options for the walk, {(chain, end): [(chain, end, scribble)]}: the links, and at each
    wiped scribble k any chain end on to any other there. A join at scribble k carries k, so the
    walk crosses it once; other joins carry 0."""
    at: dict[int, list] = {}
    for a, k in zone.items():
        if k:
            at.setdefault(k, []).append(a)
    out = {
        a: [(*b, zone.get(a, 0) if zone.get(a, 0) == zone.get(b, 0) else 0)]
        for a, b in joined.items()
    }
    for k, ends in at.items():
        for a in ends:
            out.setdefault(a, []).extend((*b, k) for b in ends)
    return out


def meets(walk: set, runs: list, found: list, idx: list, zone: dict) -> list:
    """The runs, [(chain in idx, forward)], that meet the walk's chains at a fork or a wiped
    scribble, directly or through each other."""

    def keys(chains):
        nodes = {("n", n) for i in chains for n in found[i].nodes}
        return nodes | {("z", zone.get((i, e), 0)) for i in chains for e in (0, 1)} - {("z", 0)}

    seen, todo, out = keys(walk), list(runs), []
    while met := [r for r in todo if keys([idx[j] for j, _ in r]) & seen]:
        for r in met:
            seen |= keys([idx[j] for j, _ in r])
            todo.remove(r)
        out += met
    return out
