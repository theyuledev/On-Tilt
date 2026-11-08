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
