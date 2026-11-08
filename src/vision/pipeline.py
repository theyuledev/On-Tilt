import math
import time
from typing import NamedTuple

import cv2
import numpy as np

from src.contracts import BallState
from src.vision.calib import CALIB_PATH, load_calib
from src.vision.markers import BOARD_PATH, board_pose, detector, find, load_board, relative_tilt
from src.vision.plate_frame import ball_on_plate, plate_to_px
from src.vision.plate_pose import square
from src.vision.px_to_mm import to_mm, with_velocity
from src.vision.tracker import BallPx, BallTracker


class Observation(NamedTuple):
    ball: BallState
    ball_px: BallPx
    tilt_x_deg: float  # nan until the markers give a pose
    tilt_y_deg: float
    occluded: bool
    t_read: float
    ms: float
    plate_corners_px: np.ndarray | None = None


def _pose(corners, board, lens, guess, n):
    """board_pose, or None when the solve throws or comes back non-finite."""
    try:
        pose = board_pose(corners, board, lens, guess, n)
    except cv2.error:
        return None
    if pose is None or not all(np.isfinite(v).all() for v in pose):
        return None
    return pose


def _check_ids(boards, cfg: dict) -> None:
    for name, board in zip(("plate", "base"), boards, strict=True):
        want = sorted(cfg[f"{name}_marker_ids"])
        if sorted(board) != want:
            print(f"board file {name} ids {sorted(board)} differ from settings {want}")


class Pipeline:
    """One step per frame on the grab thread; the loop just reads latest().
    Without a lens or the boards it only tracks the ball: tilt nan, never occluded."""

    def __init__(self, cfg: dict, lens: dict | None = None, boards=None):
        self.cfg = cfg["vision"]
        self.r_ball = cfg["rig"]["ball_d_mm"] / 2
        self.corners_mm = square(cfg["rig"]["plate_inner_mm"])[:, :2]
        self.tracker = BallTracker(self.cfg)
        self.history: list[BallState] = []
        self.last: Observation | None = None
        if lens is None and CALIB_PATH.exists():
            lens = load_calib(CALIB_PATH)
        if boards is None and BOARD_PATH.exists():
            boards = load_board(BOARD_PATH)
        self.lens, self.boards = lens, boards
        self.det = detector(self.cfg) if lens is not None and boards is not None else None
        if self.det is not None:
            _check_ids(boards, self.cfg)
        self.plate_pose = None  # last frame's (rvec, tvec), the next solve's guess
        self.base_pose = None  # kept once seen: the table does not move

    def latest(self) -> Observation | None:
        return self.last

    def _solve(self, gray: np.ndarray) -> None:
        corners = find(self.det, gray)
        n = self.cfg["marker_min_count"]
        self.plate_pose = _pose(corners, self.boards[0], self.lens, self.plate_pose, n)
        base = _pose(corners, self.boards[1], self.lens, self.base_pose, n)
        self.base_pose = self.base_pose if base is None else base

    def _tilt(self) -> tuple[float, float]:
        r_plate = cv2.Rodrigues(self.plate_pose[0])[0]
        r_base = None if self.base_pose is None else cv2.Rodrigues(self.base_pose[0])[0]
        return relative_tilt(r_plate, r_base)

    def step(self, image: np.ndarray, t_read: float) -> Observation:
        if self.last is not None and self.last.t_read == t_read:
            return self.last
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.det is not None:
            self._solve(gray)
        quad = None
        if self.plate_pose is not None:
            quad = plate_to_px(self.corners_mm, self.lens, *self.plate_pose)
        ball_px = self.tracker.update(gray, t_read)
        if ball_px.valid and self.last is not None and not self.last.ball_px.valid:
            self.history = []  # relock: the velocity must not bridge the gap
        xy = None
        if self.plate_pose is not None:
            xy = ball_on_plate(ball_px[:2], self.lens, *self.plate_pose, self.r_ball)
        if xy is None or not np.isfinite(xy).all():
            self.plate_pose, quad, xy = None, None, to_mm(ball_px, self.cfg)
        ball = with_velocity(*xy, ball_px, self.history, self.cfg)
        self.history = [*self.history, ball][-self.cfg["velocity_span_frames"] :]
        tilt = (math.nan, math.nan) if self.plate_pose is None else self._tilt()
        ms = (time.perf_counter() - t0) * 1e3
        occluded = self.det is not None and self.plate_pose is None
        self.last = Observation(ball, ball_px, *tilt, occluded, t_read, ms, quad)
        return self.last
