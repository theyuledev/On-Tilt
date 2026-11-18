"""The scripted player the edge guard tests lap the square maze with: chases a point ahead along
the segment it is on, moving to the next within capture mm of its end."""

from src.modes.path import Point, clip, project, seg_len, walk


def advance(path: list, seg: int, p: Point, capture: float) -> int:
    """Step the segment on while p's projection sits within capture mm of the segment's end."""
    while seg < len(path) - 2:
        a, b = path[seg], path[seg + 1]
        if seg_len(a, b) - project(p, a, b) > capture:
            break
        seg += 1
    return seg


def target(path: list, seg: int, p: Point, ahead: float, capture: float) -> tuple[int, Point]:
    """Segment to follow from p and the point to chase, at most ahead mm away from p."""
    if len(path) < 2:
        return 0, clip(p, (path[0][0], path[0][1]), ahead)
    seg = advance(path, seg, p, capture)
    goal = walk(path, seg, project(p, path[seg], path[seg + 1]), ahead)
    return seg, clip(p, goal, ahead)
