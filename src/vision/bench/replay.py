"""The tracker run over the pictures of a recorder folder and scored against their labels.
The pictures lie a second apart, so there is no track to follow: each one gets a find from
nothing, and where the ball lay at rest also one step of a tracker locked onto the label."""

from collections import Counter
from typing import NamedTuple

import cv2
import numpy as np

from src.vision.balls import Box
from src.vision.bench import score
from src.vision.bench.rec import Recording, Shot
from src.vision.bench.truth import UNREADABLE, Label, off_path
from src.vision.pipeline import Pipeline
from src.vision.px_to_mm import plate_mm
from src.vision.tracker import Background, BallPx

NO_ROW = "no row in the label file"
WARMING = "background not learned yet"
NO_BOARD = "board not found"
MOVING = "ball moving"
NO_TONE = "dark or bright ball not known"


class Scored(NamedTuple):
    frames: int
    skipped: Counter  # pictures with no score, by reason
    cold: list[score.Mark]  # found from nothing, one a scored picture
    locked: list[score.Mark]  # one step locked on the label, balls at rest only
    unlocked: Counter  # scored pictures the locked step was not run on, by reason


class Fed(Background):
    """Sampled with the labelled ball kept out, not the tracker's own answer, and never on a
    picture without a label: a ball the tracker misses must not soak into the background."""

    keep: Box | None = None

    def update(self, gray: np.ndarray, t_s: float, keep: Box | None, dark: bool) -> None:
        if self.keep is not None:
            super().update(gray, t_s, self.keep, dark)


def _around(label: Label) -> Box:
    """The margin the tracker keeps around a ball, around the whole way this one went."""
    d = 2 * label.r_px
    (ax, ay), (bx, by) = label.a, label.b
    return int(min(ax, bx) - d), int(min(ay, by) - d), int(max(ax, bx) + d), int(max(ay, by) + d)


def _mark(pipe: Pipeline, ball: BallPx, label: Label, edge_mm: float) -> score.Mark:
    a, b = (plate_mm(*p, pipe.cfg) for p in (label.a, label.b))
    where = score.regions((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, pipe.half_mm, edge_mm)
    # a coasting tracker repeats its guess: only a blob found in this picture is an answer
    found = (ball.x_px, ball.y_px) if ball.valid and pipe.tracker.misses == 0 else None
    verdict = score.judge(found, label)
    still = verdict == score.CORRECT and label.at_rest
    return score.Mark(where, verdict, off_path(plate_mm(*found, pipe.cfg), a, b) if still else None)


def _one(pipe: Pipeline, shot: Shot, label: Label | str, edge_mm: float, out: Scored) -> None:
    image = cv2.imread(str(shot.path))
    if image is None:
        out.skipped[UNREADABLE] += 1
        return
    fed, t_s = pipe.tracker.background, shot.snap["t_s"]
    known = isinstance(label, Label)
    fed.keep = _around(label) if known else None
    # asked before the step: the tracker looks first and samples after
    warm = fed.level is not None
    pipe.tracker.reset()
    seen = pipe.step(image, t_s)
    if not known:
        out.skipped[label] += 1
    elif not warm:
        out.skipped[WARMING] += 1
    elif seen.occluded:
        out.skipped[NO_BOARD] += 1
    else:
        out.cold.append(_mark(pipe, seen.ball_px, label, edge_mm))
        if not label.at_rest:
            # locked on the label the search starts on the truth itself; live it starts on a
            # guess a frame old, so a moving ball would look better here than it is
            out.unlocked[MOVING] += 1
        elif not pipe.tracker.cfg["ball_dark"]:
            out.unlocked[NO_TONE] += 1
        else:
            pipe.tracker.seed(*label.a, label.r_px, True, t_s)
            out.locked.append(_mark(pipe, pipe.tracker.update(image, t_s), label, edge_mm))


def run(recording: Recording, labels: dict[str, Label | str], cfg: dict) -> Scored:
    """One tracker a stretch, its background and board starting from nothing as they do live."""
    out = Scored(recording.frames, Counter(recording.skipped), [], [], Counter())
    pipe, stretch = None, None
    for shot in recording.shots:
        if shot.stretch != stretch:
            stretch = shot.stretch
            pipe = Pipeline(cfg)
            pipe.tracker.background = Fed(pipe.tracker.cfg)
        _one(pipe, shot, labels.get(shot.path.name, NO_ROW), cfg["bench"]["edge_mm"], out)
    return out


def total(parts: list[Scored]) -> Scored:
    return Scored(
        sum(p.frames for p in parts),
        sum((p.skipped for p in parts), Counter()),
        [m for p in parts for m in p.cold],
        [m for p in parts for m in p.locked],
        sum((p.unlocked for p in parts), Counter()),
    )
