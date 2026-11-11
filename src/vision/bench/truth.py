"""Where the ball is in each picture of a recorder folder, for scoring a tracker against.
The recorded position is the live tracker's own word, so it only becomes a label with something
else behind it; a picture with nothing behind it gets a reason instead."""

import json
import math
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from src.vision.bench import rec
from src.vision.bench.second import disc
from src.vision.bench.track import Arrivals, Track, arrivals, holding, tracks

ARRIVAL = "arrived at a driver target"
SECOND = "second detector agrees"
STILL = "still between two sightings"
UNSEEN = "ball not seen"
ALONE = "nothing backs the recorded ball"
DISPUTED = "detectors disagree"
UNREADABLE = "picture unreadable"

Px = tuple[float, float]
Second = Callable[[rec.Shot], Px | None]


class Label(NamedTuple):
    a: Px  # the ball at the snapshot the picture is named after
    b: Px  # and at the next one: in the picture it lies between the two
    r_px: float
    support: str
    at_rest: bool
    second: bool | None  # what the second detector said, where it ran


def _off_path(p: Px, a: Px, b: Px) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    span = dx * dx + dy * dy
    u = 0.0 if span == 0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span
    u = min(max(u, 0.0), 1.0)
    return math.hypot(p[0] - a[0] - u * dx, p[1] - a[1] - u * dy)


def _speed(snap: dict) -> float:
    return math.hypot(snap["ball"]["vx_mm_s"], snap["ball"]["vy_mm_s"])


def _at_rest(shot: rec.Shot, found: list[Track], bench: dict) -> bool:
    k, n = shot.snap, shot.after
    if max(_speed(k), _speed(n)) >= bench["rest_mm_s"]:
        return False
    since = k["t_s"] - bench["rest_s"]
    i = holding(found, since, n["t_s"])
    if i is None:
        return False
    t, xy = found[i]
    seen = xy[(t >= since) & (t <= n["t_s"])]
    return float(np.hypot(*(seen - seen[-1]).T).max()) < bench["rest_mm"]


def _seen(
    shot: rec.Shot, found: list[Track], arrived: set[int], bench: dict, second: Second | None
) -> Label | str:
    k, n = shot.snap, shot.after
    if not all(s["ball"]["valid"] and s["ball_px"] for s in (k, n)):
        return UNSEEN
    (ax, ay, r), (bx, by, _) = k["ball_px"], n["ball_px"]
    agrees = None
    if second is not None:
        at = second(shot)
        if at is None:
            return UNREADABLE
        agrees = _off_path(at, (ax, ay), (bx, by)) <= r
    if holding(found, k["t_s"], n["t_s"]) in arrived:
        support = ARRIVAL
    elif agrees:
        support = SECOND
    else:
        return ALONE if agrees is None else DISPUTED
    return Label((ax, ay), (bx, by), r, support, _at_rest(shot, found, bench), agrees)


def _resting(shots: list[rec.Shot], seen: list, order: range, stretch: int) -> int | None:
    """The nearest sighting that way, if it is a labelled ball at rest."""
    for j in order:
        if shots[j].stretch != stretch:
            return None
        if seen[j] != UNSEEN:
            return j if isinstance(seen[j], Label) and seen[j].at_rest else None
    return None


def _still(i: int, shots: list[rec.Shot], seen: list, bench: dict) -> Label | str:
    """A ball the live tracker missed, resting at one place in the labelled pictures before and
    after: it lay there all along."""
    before = _resting(shots, seen, range(i - 1, -1, -1), shots[i].stretch)
    after = _resting(shots, seen, range(i + 1, len(shots)), shots[i].stretch)
    if before is None or after is None:
        return UNSEEN
    p, q = shots[before].snap["ball"], shots[after].snap["ball"]
    if math.hypot(p["x_mm"] - q["x_mm"], p["y_mm"] - q["y_mm"]) >= bench["rest_mm"]:
        return UNSEEN
    at = seen[before].a
    return Label(at, at, seen[before].r_px, STILL, True, None)


def labels(
    shots: list[rec.Shot],
    found: list[Track],
    arrived: set[int],
    bench: dict,
    second: Second | None = None,
) -> list[Label | str]:
    """One label or one reason per picture."""
    seen = [_seen(shot, found, arrived, bench, second) for shot in shots]
    return [_still(i, shots, seen, bench) if got == UNSEEN else got for i, got in enumerate(seen)]


def by_disc(cfg: dict) -> Second:
    """The second detector at the ball size the settings give, never the tracker's."""
    r_px = cfg["rig"]["ball_d_mm"] / 2 * cfg["vision"]["px_per_mm"]

    def find(shot: rec.Shot) -> Px | None:
        gray = cv2.imread(str(shot.path), cv2.IMREAD_GRAYSCALE)
        return None if gray is None else disc(gray, [r_px])[:2]

    return find


def _jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def build(folder: Path, cfg: dict, second: Second | None = None):
    """The folder's pictures, a label or a reason for each, and how its trials ended."""
    bench, vision = cfg["bench"], cfg["vision"]
    recording = rec.read(folder, vision["bg_period_s"])
    # longer than the tracker coasts, and it could have lost the ball and found something else
    hole_s = vision["lost_after_misses"] / cfg["rig"]["camera_fps"]
    found = tracks(_jsonl(folder / "run.jsonl"), bench["top_speed_mm_s"], hole_s)
    trials = folder / "trials.jsonl"
    arrived = arrivals(_jsonl(trials) if trials.exists() else [], found, bench["at_target_mm"])
    got = labels(recording.shots, found, arrived.tracks, bench, second)
    return recording, got, arrived


def write(path: Path, shots: list[rec.Shot], got: list[Label | str]) -> None:
    with path.open("w") as f:
        for shot, g in zip(shots, got, strict=True):
            row = {"skip": g} if isinstance(g, str) else g._asdict()
            f.write(json.dumps({"frame": shot.path.name, **row}) + "\n")


def describe(name: str, recording: rec.Recording, got: list[Label | str], arrived: Arrivals) -> str:
    """Labels by what backs them and every picture without one by its reason."""
    marked = [g for g in got if isinstance(g, Label)]
    by = Counter(g.support for g in marked)
    checked = [g.second for g in marked if g.support == ARRIVAL and g.second is not None]
    skipped = recording.skipped + Counter(g for g in got if isinstance(g, str))
    trials = arrived.trials

    def listed(c: Counter) -> str:
        return ", ".join(f"{n} {what}" for what, n in c.most_common())

    lines = [
        f"{name}: {len(marked)} of {recording.frames} frames labelled"
        + (f" ({listed(by)})" if by else "")
        + f", {sum(g.at_rest for g in marked)} at rest",
    ]
    if checked:
        lines.append(
            f"  second detector agrees on {sum(checked)} of {len(checked)} frames that {ARRIVAL}"
        )
    if trials:
        lines.append(
            f"  {sum(trials.values())} trials: {listed(trials)}; "
            f"stuck not recorded in {arrived.stuck_unknown}"
        )
    why = f": {listed(skipped)}" if skipped else ""
    lines.append(f"  skipped {sum(skipped.values())} of {recording.frames} frames{why}")
    return "\n".join(lines)
