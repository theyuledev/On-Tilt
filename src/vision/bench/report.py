"""The scores of a replay as text: per board region how many answers were correct, a wrong
lock or lost, and the line of what was skipped, which is always there."""

from collections import Counter

import numpy as np

from src.vision.bench.replay import Scored
from src.vision.bench.score import CORRECT, LOST, REGIONS, WRONG, Mark

HEAD = (
    f"    {'region':<14}{'frames':>6}  {CORRECT:>12}  {WRONG:>12}  {LOST:>12}"
    "  off at rest mm (median, p95)"
)


def _share(n: int, of: int) -> str:
    return f"{n / of:.1%}" if of else "-"


def _listed(counts: Counter) -> str:
    return ", ".join(f"{n} {what}" for what, n in counts.most_common())


def _rows(marks: list[Mark]) -> list[str]:
    out = [HEAD]
    for region in REGIONS:
        here = [m for m in marks if region in m.regions]
        n = len(here)
        c, w, lost = (sum(m.verdict == v for m in here) for v in (CORRECT, WRONG, LOST))
        off = [m.off_mm for m in here if m.off_mm is not None]
        spread = f"{np.median(off):.2f}, {np.percentile(off, 95):.2f}" if off else "-"
        out.append(
            f"    {region:<14}{n:>6}  {c:>5} {_share(c, n):>6}  {w:>5} {_share(w, n):>6}"
            f"  {lost:>5} {_share(lost, n):>6}  {spread}"
        )
    return out


def table(name: str, got: Scored) -> str:
    skipped = sum(got.skipped.values())
    why = f": {_listed(got.skipped)}" if skipped else ""
    left = sum(got.unlocked.values())
    why_left = f": {_listed(got.unlocked)}" if left else ""
    return "\n".join(
        [
            f"{name}: {len(got.cold)} of {got.frames} frames scored",
            f"  skipped {skipped} of {got.frames} frames{why}",
            "  found from nothing",
            *_rows(got.cold),
            f"  one step locked on a ball at rest, not run on {left} scored frames{why_left}",
            *_rows(got.locked),
        ]
    )
