"""The truth labels of one recorder folder: built from its log, its trials and its pictures,
written one row a picture."""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import cv2

from src.vision.bench import rec
from src.vision.bench.second import disc
from src.vision.bench.track import Arrivals, arrivals, tracks
from src.vision.bench.truth import ARRIVAL, Label, Second, labels


class Built(NamedTuple):
    recording: rec.Recording
    labels: list[Label | str]  # one per picture of the recording, a label or the reason for none
    arrived: Arrivals
    torn: int  # log and trial lines that would not parse


def by_disc(cfg: dict) -> Second:
    """The second detector at the ball size the settings give, never the tracker's."""
    r_px = cfg["rig"]["ball_d_mm"] / 2 * cfg["vision"]["px_per_mm"]

    def find(shot: rec.Shot) -> tuple[float, float, float] | None:
        gray = cv2.imread(str(shot.path), cv2.IMREAD_GRAYSCALE)
        return None if gray is None else disc(gray, [r_px])

    return find


def _jsonl(path: Path, left_out: Counter) -> Iterator[dict]:
    """The rows of one of the folder's logs, none when the run never wrote it."""
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                left_out["torn"] += 1


def build(folder: Path, cfg: dict, second: Second | None = None) -> Built:
    bench, vision = cfg["bench"], cfg["vision"]
    recording = rec.read(folder, vision["bg_period_s"])
    left_out: Counter = Counter()
    # longer than the tracker coasts, and it could have lost the ball and found something else
    hole_s = vision["lost_after_misses"] / cfg["rig"]["camera_fps"]
    found = tracks(_jsonl(folder / "run.jsonl", left_out), bench["top_speed_mm_s"], hole_s)
    trials = _jsonl(folder / "trials.jsonl", left_out)
    arrived = arrivals(trials, found, bench["at_target_mm"])
    got = labels(recording.shots, found, arrived.tracks, bench, second)
    return Built(recording, got, arrived, left_out["torn"])


def write(path: Path, built: Built) -> None:
    with path.open("w") as f:
        for shot, got in zip(built.recording.shots, built.labels, strict=True):
            row = {"skip": got} if isinstance(got, str) else got._asdict()
            f.write(json.dumps({"frame": shot.path.name, **row}) + "\n")


def read(path: Path) -> dict[str, Label | str]:
    """What write() wrote, by picture name."""
    out: dict[str, Label | str] = {}
    with path.open() as f:
        for row in map(json.loads, f):
            name = row.pop("frame")
            if "skip" in row:
                out[name] = row["skip"]
            else:
                out[name] = Label(**{**row, "a": tuple(row["a"]), "b": tuple(row["b"])})
    return out


def _listed(counts: Counter) -> str:
    return ", ".join(f"{n} {what}" for what, n in counts.most_common())


def describe(name: str, built: Built) -> str:
    """Labels by what backs them and every picture without one by its reason."""
    frames, trials = built.recording.frames, built.arrived.trials
    marked = [g for g in built.labels if isinstance(g, Label)]
    by = Counter(g.support for g in marked)
    checked = [g.second for g in marked if g.support == ARRIVAL and g.second is not None]
    skipped = built.recording.skipped + Counter(g for g in built.labels if isinstance(g, str))
    lines = [
        f"{name}: {len(marked)} of {frames} frames labelled"
        + (f" ({_listed(by)})" if by else "")
        + f", {sum(g.at_rest for g in marked)} at rest",
    ]
    if checked:
        lines.append(
            f"  second detector agrees on {sum(checked)} of {len(checked)} frames that {ARRIVAL}"
        )
    if trials:
        lines.append(
            f"  {sum(trials.values())} trials: {_listed(trials)}; "
            f"stuck not recorded in {built.arrived.stuck_unknown}"
        )
    if built.torn:
        lines.append(f"  {built.torn} torn log lines left out")
    why = f": {_listed(skipped)}" if skipped else ""
    lines.append(f"  skipped {sum(skipped.values())} of {frames} frames{why}")
    return "\n".join(lines)
