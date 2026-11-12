"""A tracker's answer on one picture against the label of that picture, and where on the board
the ball lay, so the answers can be counted by region."""

from typing import NamedTuple

from src.vision.bench.truth import Label, Px, off_path

CORRECT, WRONG, LOST = "correct", "wrong lock", "lost"
MIDDLE, SIDES, CORNERS, OFF = "middle", "sides", "corners", "off the board"
EDGE, ALL = "edge band", "all"
REGIONS = (MIDDLE, SIDES, CORNERS, OFF, EDGE, ALL)


class Mark(NamedTuple):
    regions: tuple[str, ...]
    verdict: str
    off_mm: float | None  # a correct answer on a ball at rest: how far from the label it lay


def judge(found: Px | None, label: Label) -> str:
    """Correct when the answer's centre lies inside the real ball, wherever the ball was on its
    way between the two snapshots the picture was taken between."""
    if found is None:
        return LOST
    return CORRECT if off_path(found, label.a, label.b) <= label.r_px else WRONG


def regions(x_mm: float, y_mm: float, half_mm: float, edge_mm: float) -> tuple[str, ...]:
    """The board cut 3 x 3 into its middle, four sides and four corners. A ball within edge_mm
    of an edge counts in the edge band as well, on top of its side or corner."""
    reach = max(abs(x_mm), abs(y_mm))
    if reach > half_mm:
        return OFF, ALL
    third = (MIDDLE, SIDES, CORNERS)[sum(abs(v) > half_mm / 3 for v in (x_mm, y_mm))]
    return (third, EDGE, ALL) if half_mm - reach <= edge_mm else (third, ALL)
