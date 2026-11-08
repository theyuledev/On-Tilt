"""Pure geometry and pace for a ball following a polyline of (x, y) points in mm."""

import math

Point = tuple[float, float]
STILL = (0.0, 0.0)


def seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def unit(a: Point, b: Point) -> Point:
    length = seg_len(a, b)
    return ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def project(p: Point, a: Point, b: Point) -> float:
    """Distance along a->b of the point nearest p, clamped onto the segment."""
    length = seg_len(a, b)
    if length == 0:
        return 0.0
    s = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / length
    return min(max(s, 0.0), length)


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


def corner_speed(path: list, i: int, capture: float, acc: float) -> float:
    """Speed the ball can carry round path[i] at acc mm/s^2 sideways; 0 at either end."""
    if i <= 0 or i >= len(path) - 1:
        return 0.0
    t0, t1 = unit(path[i - 1], path[i]), unit(path[i], path[i + 1])
    turn = math.acos(min(max(t0[0] * t1[0] + t0[1] * t1[1], -1.0), 1.0))
    if turn == 0:
        return math.inf
    # the segment switches capture mm before the corner, so the ball rounds it on that fillet
    return math.sqrt(acc * capture / math.tan(turn / 2))


def pace(s: float, stop: float, v_in: float, v_out: float, top: float, acc: float):
    """Path speed and acceleration at s: up from v_in, cruise at top, down to v_out by stop."""
    up = math.sqrt(v_in**2 + 2 * acc * s)
    down = math.sqrt(v_out**2 + 2 * acc * max(stop - s, 0.0))
    if top <= min(up, down):
        return top, 0.0
    return (up, acc) if up < down else (down, -acc)


def guide(path: list, seg: int, p: Point, c: dict) -> tuple[int, Point, Point, Point]:
    """Segment to follow, the goal for p (its foot on the path, the pull clipped to the lookahead),
    the path velocity to hold and the feed forward tilt in degrees."""
    ahead, capture = c["path_lookahead_mm"], c["path_capture_mm"]
    if len(path) < 2:
        return 0, clip(p, path[0], ahead), STILL, STILL
    seg = advance(path, seg, p, capture)
    a, b = path[seg], path[seg + 1]
    length, s = seg_len(a, b), project(p, a, b)
    if seg == len(path) - 2 and length - s <= capture:
        return seg, clip(p, b, ahead), STILL, STILL
    t = unit(a, b)
    foot = (a[0] + s * t[0], a[1] + s * t[1])
    near = clip(foot, p, ahead)
    goal = (p[0] - near[0] + foot[0], p[1] - near[1] + foot[1])
    acc = c["path_accel_mm_s2"]
    v_in, v_out = corner_speed(path, seg, capture, acc), corner_speed(path, seg + 1, capture, acc)
    v, dv = pace(s, length - capture, v_in, v_out, c["path_speed_mm_s"], acc)
    # full pace within capture of the line, none while the ball is still a lookahead away
    k = min(max((ahead - seg_len(near, foot)) / (ahead - capture), 0.0), 1.0)
    # solid ball rolling, as in the sim: 5/7 g per radian of tilt
    lean = math.degrees(k * dv / (5 / 7 * c["sim_g_mm_s2"]))
    return seg, goal, (k * v * t[0], k * v * t[1]), (lean * t[0], lean * t[1])
