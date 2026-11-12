"""Pure geometry for a setpoint that walks a polyline of (x, y) points in mm."""

import math

Point = tuple[float, float]


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


def advance(path: list, seg: int, p: Point, capture: float) -> int:
    """Step the segment on while p's projection sits within capture mm of the segment's end."""
    while seg < len(path) - 2:
        a, b = path[seg], path[seg + 1]
        if seg_len(a, b) - project(p, a, b) > capture:
            break
        seg += 1
    return seg


def clip(p: Point, goal: Point, reach: float) -> Point:
    """goal pulled back to within reach mm of p, so a far ball joins at cruising speed."""
    d = seg_len(p, goal)
    if d <= reach:
        return goal
    f = reach / d
    return (p[0] + f * (goal[0] - p[0]), p[1] + f * (goal[1] - p[1]))


def target(path: list, seg: int, p: Point, ahead: float, capture: float) -> tuple[int, Point]:
    """Segment to follow from p and the point to chase, at most ahead mm away from p."""
    if len(path) < 2:
        return 0, clip(p, (path[0][0], path[0][1]), ahead)
    seg = advance(path, seg, p, capture)
    goal = walk(path, seg, project(p, path[seg], path[seg + 1]), ahead)
    return seg, clip(p, goal, ahead)


def is_open(path: list) -> bool:
    """Two ends to go between: at least two points and the last is not the first again."""
    return len(path) > 1 and tuple(path[0]) != tuple(path[-1])


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
