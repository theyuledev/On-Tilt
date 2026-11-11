"""Corridor path through a free-space mask, and the polyline and maze file made from it."""

import heapq
import math

import cv2
import numpy as np

STEPS = [(dy, dx, math.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def clearance(free: np.ndarray) -> np.ndarray:
    edged = np.uint8(free)
    edged[[0, -1], :] = edged[:, [0, -1]] = 0
    return cv2.distanceTransform(edged, cv2.DIST_L2, 5)


def _snap(clear: np.ndarray, p) -> int:
    ys, xs = np.nonzero(clear)
    k = np.argmin((xs - p[0]) ** 2 + (ys - p[1]) ** 2)
    return int(ys[k]) * clear.shape[1] + int(xs[k])


def shortest_path(clear, a, b, cap_px: float) -> list[tuple[int, int]] | None:
    """(x, y) px from a to b where clear > 0; steps near walls cost more so it keeps central."""
    if not np.any(clear > 0):
        return None
    h, w = clear.shape
    cost = (1.0 / np.minimum(np.maximum(clear, 1e-3), cap_px)).ravel().tolist()
    ok = (clear > 0).ravel().tolist()
    src, dst = _snap(clear, a), _snap(clear, b)
    best, prev, heap = {src: 0.0}, {}, [(0.0, src)]
    while heap:
        d, i = heapq.heappop(heap)
        if i == dst:
            break
        if d > best[i]:
            continue
        y, x = divmod(i, w)
        for dy, dx, step in STEPS:
            if not (0 <= y + dy < h and 0 <= x + dx < w):
                continue
            j = i + dy * w + dx
            if dx and dy and (not ok[i + dx] or not ok[i + dy * w]):
                continue
            if ok[j]:
                nd = d + step * (cost[i] + cost[j]) / 2
                if nd < best.get(j, math.inf):
                    best[j], prev[j] = nd, i
                    heapq.heappush(heap, (nd, j))
    else:
        return None
    out = [dst]
    while out[-1] != src:
        out.append(prev[out[-1]])
    return [(i % w, i // w) for i in reversed(out)]


def polyline(pts: np.ndarray, tol: float, step: float) -> list[tuple[float, float]]:
    """Corners kept within tol, long runs split so no two points are more than step apart."""
    keep = cv2.approxPolyDP(np.float32(pts).reshape(-1, 1, 2), tol, False).reshape(-1, 2)
    out = [tuple(keep[0])]
    for p, q in zip(keep[:-1], keep[1:], strict=True):
        n = max(1, math.ceil(math.dist(p, q) / step))
        out += [tuple(p + (q - p) * k / n) for k in range(1, n + 1)]
    return [(float(x), float(y)) for x, y in out]


def length(pts) -> float:
    return sum(math.dist(p, q) for p, q in zip(pts[:-1], pts[1:], strict=True))


def checkpoints(points, n: int, min_mm: float) -> list[int]:
    """Up to n gates spread along the line, the last on the finish, each at least 2 points and
    min_mm from the others and from the start."""
    steps = [math.dist(p, q) for p, q in zip(points[:-1], points[1:], strict=True)]
    along = np.cumsum([0.0, *steps])
    gates = [len(points) - 1]
    for k in range(n - 1, 0, -1):
        i = int(np.argmin(abs(along - along[-1] * k / n)))
        far = all(math.dist(points[i], points[g]) >= min_mm for g in [0, *gates])
        if far and 2 <= i <= gates[0] - 2:
            gates.insert(0, i)
    return gates


def _pts(line) -> str:
    return "[" + ", ".join(f"[{x:.1f}, {y:.1f}]" for x, y in line) + "]"


def maze_toml(points, gates: list[int], walls) -> str:
    x0, y0 = points[0]
    return (
        f"points = {_pts(points)}\ncheckpoints = {gates}\nstart = [{x0:.1f}, {y0:.1f}]\n"
        f"walls = [{', '.join(_pts(w) for w in walls)}]\n"
    )
