"""Pure geometry for a setpoint that walks a polyline of (x, y) points in mm."""

import bisect
import math

Point = tuple[float, float]
HALVINGS = 10  # finds where the line leaves the straight to about 0.01 mm on 10 mm pieces


def seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def project(p: Point, a: Point, b: Point) -> float:
    """Distance along a->b of the point nearest p, clamped onto the segment."""
    length = seg_len(a, b)
    if length == 0:
        return 0.0
    s = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / length
    return min(max(s, 0.0), length)


def walk(path: list, seg: int, s: float, ahead: float) -> Point:
    """The point `ahead` mm past distance s on segment seg; the last point once the path ends."""
    while seg < len(path) - 1:
        a, b = path[seg], path[seg + 1]
        length = seg_len(a, b)
        if s + ahead <= length:
            f = (s + ahead) / length
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        ahead -= length - s
        s = 0.0
        seg += 1
    return (path[-1][0], path[-1][1])


def clip(p: Point, goal: Point, reach: float) -> Point:
    """goal pulled back to within reach mm of p, so a far ball joins at cruising speed."""
    d = seg_len(p, goal)
    if d <= reach:
        return goal
    f = reach / d
    return (p[0] + f * (goal[0] - p[0]), p[1] + f * (goal[1] - p[1]))


def loop_start(path: list) -> int | None:
    """Index of the first point the last one repeats, where the path is lapped from on; None for
    a path with two ends."""
    return next((k for k, p in enumerate(path[:-1]) if p == path[-1]), None)


def tidy(points: list) -> list:
    """points without the ones that repeat their neighbour."""
    return [p for i, p in enumerate(points) if i == 0 or p != points[i - 1]]


def lengths(path: list) -> list[float]:
    """Distance along the path at each of its points."""
    cum = [0.0]
    for a, b in zip(path, path[1:], strict=False):
        cum.append(cum[-1] + seg_len(a, b))
    return cum


def from_nearest_end(path: list, p: Point) -> list:
    """path turned round when its last point is the end nearer p."""
    return path[::-1] if seg_len(p, path[-1]) < seg_len(p, path[0]) else path


def progress(path: list, cum: list, p: Point, s: float, window: float) -> float:
    """Distance along the path of its point nearest p between s and s + window: it never goes
    back, and a later pass of the path close by is out of reach."""
    best, near = s, math.inf
    for i in range(len(path) - 1):
        lo, hi = max(s, cum[i]), min(s + window, cum[i + 1])
        if lo > hi:
            continue
        t = min(max(cum[i] + project(p, path[i], path[i + 1]), lo), hi)
        d = seg_len(p, walk(path, i, t - cum[i], 0.0))
        if d < near:
            best, near = t, d
    return best


def stretch(p: Point, goal: Point, reach: float) -> Point:
    """goal moved along the line from p until it is reach mm away, so the pull keeps its size
    however near the goal is. p itself when it sits on the goal: no direction to pull in."""
    d = seg_len(p, goal)
    if d == 0:
        return p
    f = reach / d
    return (p[0] + f * (goal[0] - p[0]), p[1] + f * (goal[1] - p[1]))


def heading(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def tangent(path: list, cum: list, s: float) -> float:
    """Heading of the path at distance s along it, of its last piece past the end."""
    i = min(bisect.bisect_right(cum, s), len(cum) - 1)
    return heading(path[i - 1], path[i])


def turned(a: float, b: float) -> float:
    """Angle from heading a to heading b, between -pi and pi."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi


def apart(p: Point, a: Point, b: Point) -> float:
    """Distance from p to the segment a->b."""
    return seg_len(p, walk([a, b], 0, project(p, a, b), 0.0)) if a != b else seg_len(p, a)


def sight(path: list, cum: list, s: float, ahead: float, tol: float) -> float:
    """How far past distance s the path stays within tol mm of the straight line from its point
    at s, up to ahead mm: the part of it a ball can be sent along in a straight line. A path
    that ends straight counts as going on straight."""
    here, end = walk(path, 0, 0.0, s), min(s + ahead, cum[-1])
    inner = [i for i, d in enumerate(cum) if s < d < end]
    ends = [cum[i] for i in inner] + [end]

    def fits(d: float, k: int) -> bool:
        to = walk(path, 0, 0.0, d)
        return all(apart(path[i], here, to) <= tol for i in inner[:k])

    lo = s
    for k, hi in enumerate(ends):
        if not fits(hi, k):
            for _ in range(HALVINGS):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if fits(mid, k) else (lo, mid)
            return lo - s
        lo = hi
    return ahead
