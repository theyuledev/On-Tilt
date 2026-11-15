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
    """Join options for the walk, {(chain, end): [(chain, end, scribble, free)]}: the links, and
    at each wiped scribble any chain end on to any other there. A join at scribble k carries k, so
    the walk crosses it once; where only two chains leave it (have one end there, not both) the
    join between them is free to turn any amount, as the ink that showed the turn is wiped and
    there is no other way on."""
    at: dict[int, list] = {}
    for a, k in zone.items():
        if k:
            at.setdefault(k, []).append(a)
    leave = {k: {a for a in ends if zone[a[0], 1 - a[1]] != k} for k, ends in at.items()}

    def tag(a, b):
        k = zone.get(a, 0) if zone.get(a, 0) == zone.get(b, 0) else 0
        return (*b, k, k > 0 and len(leave[k]) == 2 and {a, b} == leave[k])

    out = {a: [tag(a, b)] for a, b in joined.items()}
    for ends in at.values():
        for a in ends:
            out.setdefault(a, []).extend(tag(a, b) for b in ends if b != a and b != joined.get(a))
    return out
