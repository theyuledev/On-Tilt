import math
import time
from typing import NamedTuple

import numpy as np

from src.contracts import BallState
from src.vision.balls import Candidate
from src.vision.plate_find import PlateTracker, configured
from src.vision.px_to_mm import ball_state, outline_px, plate_mm, with_calib
from src.vision.tracker import BallPx, BallTracker


class Observation(NamedTuple):
    ball: BallState
    ball_px: BallPx
    tilt_x_deg: float  # nan until the markers give a pose
    tilt_y_deg: float
    occluded: bool  # plate tracking is on and the board is not found; the map stays where it was
    t_read: float
    ms: float
    outline_px: list | None = None  # the tracked board in the picture, a new list when it moved


class Pipeline:
    """One step per frame on the grab thread; the loop just reads latest()."""

    def __init__(self, cfg: dict):
        self.cfg = with_calib(cfg["vision"])
        self.half_mm = cfg["rig"]["plate_inner_mm"] / 2
        self.reach_mm = self.half_mm + cfg["rig"]["ball_d_mm"]
        self.tracker = BallTracker(self.cfg, self._on_plate)
        self.plate: PlateTracker | None = None
        if "calib" in self.cfg and self.cfg["plate_track"]:
            self.plate = PlateTracker(self.cfg, self.half_mm, configured(self.half_mm, self.cfg))
        self.outline: list | None = None
        self.outline_t = -math.inf
        self.history: list[BallState] = []
        self.last: Observation | None = None

    def latest(self) -> Observation | None:
        return self.last

    def _on_plate(self, c: Candidate) -> bool:
        """A blob more than a ball's width past the edge is not the ball. Only judged against a
        board found in this frame: a fixed map can be that far out at a corner by itself."""
        if self.plate is None or self.plate.lost:
            return True
        return max(map(abs, plate_mm(c.x_px, c.y_px, self.cfg))) <= self.reach_mm

    def _follow(self, image: np.ndarray, t_read: float) -> None:
        """Puts the map on the board as this frame shows it, so a shaking camera stays out of the
        ball's mm. The page's outline follows at its own slow rate."""
        self.plate.update(image)
        self.cfg = self.plate.mapped()
        if abs(t_read - self.outline_t) >= self.cfg["plate_outline_period_s"]:
            self.outline_t = t_read
            outline = outline_px(self.half_mm, self.cfg)
            if outline != self.outline:
                self.outline = outline

    def step(self, image: np.ndarray, t_read: float) -> Observation:
        if self.last is not None and self.last.t_read == t_read:
            return self.last
        t0 = time.perf_counter()
        if self.plate is not None:
            self._follow(image, t_read)
        ball_px = self.tracker.update(image, t_read)
        if ball_px.valid and self.last is not None and not self.last.ball_px.valid:
            self.history = []  # relock: the velocity must not bridge the gap
        ball = ball_state(ball_px, self.history, self.cfg)
        self.history = [*self.history, ball][-self.cfg["velocity_span_frames"] :]
        ms = (time.perf_counter() - t0) * 1e3
        lost = self.plate is not None and self.plate.lost
        self.last = Observation(ball, ball_px, math.nan, math.nan, lost, t_read, ms, self.outline)
        return self.last
