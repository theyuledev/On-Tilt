"""A recorder folder for offline replay: every saved picture with the snapshot it is named
after and the one that follows, under the settings the run was made with."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from statistics import median
from typing import NamedTuple

from src.io.settings import load
from src.vision.calib import load_calib

NO_STAMP = "no snapshot of its stamp"
NO_NEXT = "no snapshot right after it"


class Shot(NamedTuple):
    path: Path
    snap: dict  # the snapshot the picture is named after
    after: dict  # the next one: the recorder fetched the picture between the two
    stretch: int


class Recording(NamedTuple):
    shots: list[Shot]
    frames: int
    skipped: Counter  # pictures left out, by reason
    stretches: int
    counts: Counter  # snapshot lines: snapshots, torn, armed, seen, board lost, board not recorded


def settings(folder: Path) -> dict:
    """The checked-in settings under the folder's own local file."""
    local = folder / "settings.local.toml"
    if not local.exists():
        raise FileNotFoundError(f"{local}: not a recorder folder")
    return load(local=local)


def calibrated(cfg: dict, calib: Path) -> dict:
    """cfg with the calibration handed over inside the vision settings, so a Pipeline built
    from it reads nothing from this checkout's local files."""
    vision = cfg["vision"]
    if not vision["use_calib"]:
        return cfg
    return {**cfg, "vision": {**vision, "use_calib": False, "calib": load_calib(calib)}}


def _rows(folder: Path, counts: Counter) -> Iterator[dict]:
    with (folder / "snap.jsonl").open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                counts["torn"] += 1
                continue
            counts["snapshots"] += 1
            if row["armed"]:
                counts["armed"] += 1
                counts["seen"] += bool(row["ball"]["valid"])
            if "plate_lost" not in row:
                counts["board not recorded"] += 1
            elif row["plate_lost"]:
                counts["board lost"] += 1
            yield row


def read(folder: Path, gap_s: float) -> Recording:
    """Pairs pictures with snapshots. A step back in time or a gap over gap_s starts a new
    stretch; a picture whose next snapshot lies across one has no bracket and is left out."""
    names = {p.name: p for p in (folder / "frames").iterdir()}
    frames = len(names)
    shots: list[Shot] = []
    counts: Counter = Counter()
    skipped: Counter = Counter()
    stretches = 0
    prev = held = None
    for row in _rows(folder, counts):
        if prev is None or not 0 <= row["t_s"] - prev["t_s"] <= gap_s:
            stretches += 1
            if held is not None:
                skipped[NO_NEXT] += 1
            held = None
        if held is not None:
            shots.append(Shot(held, prev, row, stretches - 1))
        held = names.pop(f"{row['t_s']:010.3f}.jpg", None)
        prev = row
    if held is not None:
        skipped[NO_NEXT] += 1
    if names:
        skipped[NO_STAMP] += len(names)
    return Recording(shots, frames, skipped, stretches, counts)


def describe(name: str, rec: Recording) -> str:
    """What the folder holds and what a replay would leave out, the skipped line always there."""
    c = rec.counts
    seen = f"{c['seen'] / c['armed']:.1%}" if c["armed"] else "-"
    gaps = [(s.after["t_s"] - s.snap["t_s"]) * 1e3 for s in rec.shots]
    bracket = f"median {median(gaps):.0f} ms, longest {max(gaps):.0f} ms" if gaps else "-"
    why = ", ".join(f"{n} {reason}" for reason, n in rec.skipped.most_common())
    skipped = f"skipped {sum(rec.skipped.values())} of {rec.frames} frames"
    return "\n".join(
        [
            f"{name}: {rec.frames} frames, {c['snapshots']} snapshots, "
            f"{c['torn']} torn lines, {rec.stretches} stretches",
            f"  armed in {c['armed']} snapshots, ball seen in {seen} of them; board lost in "
            f"{c['board lost']}, not recorded in {c['board not recorded']}",
            f"  {len(rec.shots)} frames paired, snapshot to next {bracket}",
            f"  {skipped}: {why}" if why else f"  {skipped}",
        ]
    )
