import math
import time
from typing import NamedTuple

import numpy as np

from src.contracts import BallState
from src.vision.px_to_mm import ball_state, with_calib
from src.vision.tracker import BallPx, BallTracker


class Observation(NamedTuple):
    ball: BallState
    ball_px: BallPx
    tilt_x_deg: float  # nan until the markers give a pose
    tilt_y_deg: float
    occluded: bool
    t_read: float
    ms: float


class Pipeline:
    """One step per frame on the grab thread; the loop just reads latest()."""

    def __init__(self, cfg: dict):
        self.cfg = with_calib(cfg["vision"])
        self.tracker = BallTracker(self.cfg)
        self.history: list[BallState] = []
        self.last: Observation | None = None

    def latest(self) -> Observation | None:
        return self.last

    def step(self, image: np.ndarray, t_read: float) -> Observation:
        if self.last is not None and self.last.t_read == t_read:
            return self.last
        t0 = time.perf_counter()
        ball_px = self.tracker.update(image, t_read)
        if ball_px.valid and self.last is not None and not self.last.ball_px.valid:
            self.history = []  # relock: the velocity must not bridge the gap
        ball = ball_state(ball_px, self.history, self.cfg)
        self.history = [*self.history, ball][-self.cfg["velocity_span_frames"] :]
        ms = (time.perf_counter() - t0) * 1e3
        self.last = Observation(ball, ball_px, math.nan, math.nan, False, t_read, ms)
        return self.last
