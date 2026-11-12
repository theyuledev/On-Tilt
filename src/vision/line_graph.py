"""A drawn line's skeleton as chains between forks and loose ends, joined across gaps where the
ends point at each other, and the longest walk through them that never turns sharply."""

import math
from typing import NamedTuple

import cv2
import numpy as np

RING = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]


class Chain(NamedTuple):
    px: np.ndarray  # (n, 2) x, y in order
    nodes: tuple[int, int]
    out: tuple[np.ndarray, np.ndarray]  # unit direction leaving the chain at each end
    length: float


def runs(skel: np.ndarray) -> np.ndarray:
    """How many skeleton runs meet at each pixel: 1 at an end, 2 along, 3 or more at a fork."""
    s = np.pad(skel, 1).astype(np.uint8)
    h, w = skel.shape
    ring = [s[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dx, dy in RING]
    return sum((ring[i] == 0) & (ring[(i + 1) % 8] == 1) for i in range(8)) * skel


def _far(pts: set, start) -> tuple:
    prev, todo, last = {start: start}, [start], start
    while todo:
        last = todo.pop(0)
        for dx, dy in RING:
            p = (last[0] + dx, last[1] + dy)
            if p in pts and p not in prev:
                prev[p] = last
                todo.append(p)
    return last, prev


def _order(pts: set, ends: list) -> list:
    """The run's pixels from one end to the far end, or round it when it is a loop."""
    if not ends:
        out, seen = [next(iter(pts))], set()
        while out[-1] not in seen:
            seen.add(out[-1])
            x, y = out[-1]
            nxt = [(x + dx, y + dy) for dx, dy in RING if (x + dx, y + dy) in pts - seen]
            out.append(min(nxt, key=lambda p: abs(p[0] - x) + abs(p[1] - y)) if nxt else out[-1])
        return out[:-1]
    b, prev = _far(pts, ends[0])
    out = [b]
    while out[-1] != ends[0]:
        out.append(prev[out[-1]])
    return out


def _out(px: np.ndarray, back: int) -> tuple[np.ndarray, np.ndarray]:
    """Direction leaving each end, measured over back pixels."""
    n = len(px) - 1
    far = min(back, n)
    a, b = px[0] - px[far], px[n] - px[n - far]
    return tuple(v / max(np.linalg.norm(v), 1e-9) for v in (a, b))


def chains(skel: np.ndarray, back: int) -> list[Chain]:
    """Skeleton runs between forks; a run's end next to a fork shares that fork's node."""
    # a fork pixel's neighbours too, or two branches can still touch corner to corner
    fork = cv2.dilate(np.uint8(runs(skel) >= 3), np.ones((3, 3), np.uint8)).astype(bool) & skel
    nf, forks = cv2.connectedComponents(np.uint8(fork))
    n, lab = cv2.connectedComponents(np.uint8(skel & ~fork))
    ys, xs = np.nonzero(lab)
    groups: dict[int, set] = {}
    for x, y, k in zip(xs.tolist(), ys.tolist(), lab[ys, xs].tolist(), strict=True):
        groups.setdefault(k, set()).add((x, y))
    h, w = skel.shape
    found, free = [], nf
    for pts in groups.values():
        ends = [p for p in pts if sum((p[0] + dx, p[1] + dy) in pts for dx, dy in RING) <= 1]
        px = np.array(_order(pts, ends), float)
        nodes = []
        for x, y in (px[0], px[-1]):
            near = [
                forks[int(y) + dy, int(x) + dx]
                for dx, dy in RING
                if 0 <= int(y) + dy < h and 0 <= int(x) + dx < w
            ]
            hit = [f for f in near if f]
            nodes.append(hit[0] if hit else free)
            free += not hit
        steps = np.linalg.norm(np.diff(px, axis=0), axis=1).sum() if len(px) > 1 else 1.0
        found.append(Chain(px, tuple(nodes), _out(px, back), float(steps)))
    return found


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))


def links(found: list[Chain], reach: float, bend: float, hole=None) -> dict:
    """Chain ends joined two by two, cheapest first: the gap over reach plus the turn over bend,
    where each end may turn half of bend towards the other. Two ends both inside hole, (x, y, r)
    px where the ball hides the line, may join from any distance. No end is used twice. Gives
    {(chain, end): (chain, end)} both ways."""
    ends = [(c.px[-e], c.out[e], i, e) for i, c in enumerate(found) for e in (0, 1)]
    pairs = []
    for k, (p, t, i, e) in enumerate(ends):
        for q, u, j, f in ends[k + 1 :]:
            v = q - p
            d = max(float(np.linalg.norm(v)), 1e-9)
            hidden = (
                hole is not None and max(math.dist(p, hole[:2]), math.dist(q, hole[:2])) <= hole[2]
            )
            if i == j or (d > reach and not hidden):
                continue
            turn = 2 * max(_angle(t, v / d), _angle(u, -v / d))
            if turn <= bend:
                pairs.append((d / reach + turn / bend, (i, e), (j, f)))
    out: dict = {}
    for _, a, b in sorted(pairs, key=lambda x: x[0]):
        if a not in out and b not in out:
            out[a], out[b] = b, a
    return out


def longest_walk(found: list[Chain], joined: dict, bend: float, cost: float, starts, budget=300000):
    """The walk over chains and joins, each chain once, turning at most bend degrees, that
    covers the most drawn length less cost px per 90 degrees turned. It begins at one of
    starts, (chain, end) pairs; straightest turns are tried first and the starts share budget
    steps evenly. Gives [(chain, forward)]."""
    at: dict[int, list] = {}
    for i, c in enumerate(found):
        at.setdefault(c.nodes[0], []).append((i, 0))
        at.setdefault(c.nodes[1], []).append((i, 1))
    best, left = ([], -1.0), [0]

    def nexts(i, e):
        far = (i, 1 - e)
        out = [(j, f) for j, f in at[found[i].nodes[1 - e]] if j != i]
        return out + ([joined[far]] if far in joined else [])

    def go(path, used, total, i, e, turn):
        nonlocal best
        used.add(i)
        path.append((i, e == 0))
        total += found[i].length - cost * turn / 90
        if total > best[1]:
            best = (list(path), total)
        heading = found[i].out[1 - e]
        turns = sorted((_angle(heading, -found[j].out[f]), j, f) for j, f in nexts(i, e))
        for t, j, f in turns:
            left[0] -= 1
            if left[0] < 0:
                break
            if j not in used and t <= bend:
                go(path, used, total, j, f, t)
        path.pop()
        used.discard(i)

    for i, e in starts:
        left[0] = budget // len(starts)
        go([], set(), 0.0, i, e, 0.0)
    return best[0]


def lines(found: list[Chain], joined: dict) -> list[list[tuple[int, bool]]]:
    """The joined chains as runs end to end, [(chain, forward)], longest drawn first."""
    seen, out = set(), []
    for i in range(len(found)):
        for e in (0, 1):
            if i in seen or (i, e) in joined:
                continue
            run, at = [], (i, e)
            while at is not None and at[0] not in seen:
                seen.add(at[0])
                run.append((at[0], at[1] == 0))
                at = joined.get((at[0], 1 - at[1]))
            out.append(run)
    return sorted(out, key=lambda r: -sum(found[i].length for i, _ in r))


def drawn(found: list[Chain], run) -> float:
    return sum(found[i].length for i, _ in run)


def walk_px(found: list[Chain], walk) -> np.ndarray:
    parts = [found[i].px if fwd else found[i].px[::-1] for i, fwd in walk]
    return np.vstack(parts) if parts else np.zeros((0, 2))
