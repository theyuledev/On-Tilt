import time
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.control.pulse_map import center
from src.io.camera import ClipSource
from src.io.ports import Frame
from src.io.settings import load
from src.loop import Loop
from src.modes import LEVEL
from src.safety import limits
from src.vision.pipeline import Pipeline
from tests.replay.test_maze_clip import CLIP as MAZE
from tests.replay.test_maze_clip import EMPTY_FROM_S, REST_UNTIL_S
from tests.replay.test_pipeline_clip import CLIP as SQUASH
from tests.replay.test_pipeline_clip import END, START

pytestmark = pytest.mark.replay


def config():
    # module fixtures run before the conftest one that hides settings.local.toml
    return load(local=Path("/dev/null"))


class JpgFrames:
    """The maze jpgs as a camera, read like test_maze_clip and run through process."""

    def __init__(self, folder: Path, process):
        paths = sorted(folder.glob("f_*.jpg"), key=lambda p: float(p.stem[2:]))
        self.paths, prev = [], None
        for p in paths:
            img = cv2.imread(str(p))
            # back to back copies are one camera frame saved twice
            if prev is None or not np.array_equal(img, prev):
                self.paths.append(p)
            prev = img
        self.stamps = [float(p.stem[2:]) for p in self.paths]
        self.process, self.result, self.i = process, None, 0

    def read(self) -> Frame | None:
        if self.i == len(self.paths):
            return None
        img = cv2.imread(str(self.paths[self.i]))
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2)
        frame = Frame(gray, self.stamps[self.i])
        self.i += 1
        self.result = self.process(*frame)
        return frame

    def latest(self):
        return self.result

    def close(self) -> None:
        pass


def run(cfg, frames, stamps):
    """Balance armed on the sim plate, one tick per frame at the frame's own time."""
    loop = Loop(cfg, frames=frames)
    loop.mode = "balance"
    loop.arm()
    out, prev, clock = [], stamps[0], [0.0]
    with pytest.MonkeyPatch.context() as mp:
        # keys stamp with time.monotonic; the page's heartbeat keeps them fresh while armed
        mp.setattr(time, "monotonic", lambda: clock[0])
        for t in stamps:
            clock[0] = t
            loop.keys.set(0.0, 0.0)
            rec = loop.tick(t, t - prev)
            out.append((t, frames.latest(), rec, loop.snapshot()))
            prev = t
    frames.close()
    return out


@pytest.fixture(scope="module")
def squash():
    if not SQUASH.exists():
        pytest.skip(f"{SQUASH.name} not on this machine")
    cfg = config()
    stamps = [float(t) for t in SQUASH.with_suffix(".csv").read_text().split()][START:END]
    frames = ClipSource(SQUASH, start=START, process=Pipeline(cfg).step)
    return cfg, run(cfg, frames, stamps)


@pytest.fixture(scope="module")
def maze():
    if not MAZE.exists():
        pytest.skip(f"{MAZE.name} not on this machine")
    cfg = config()
    frames = JpgFrames(MAZE, Pipeline(cfg).step)
    ticks = run(cfg, frames, frames.stamps)
    t0 = frames.stamps[0]
    return cfg, [(t - t0, obs, rec, snap) for t, obs, rec, snap in ticks]


def test_loop_ball_is_the_newest_observation_and_the_page_gets_its_pixels(squash):
    _, ticks = squash
    assert any(obs.ball.valid for _, obs, _, _ in ticks)
    for t, obs, rec, snap in ticks:
        assert obs.t_read == t and rec.ball == obs.ball
        px = obs.ball_px
        assert snap["ball_px"] == ([px.x_px, px.y_px, px.r_px] if obs.ball.valid else None)


def test_lost_ball_levels_and_pulses_stay_in_limits_and_slew(squash):
    cfg, ticks = squash
    rig = cfg["rig"]
    assert all(rec.plate_cmd == LEVEL for _, obs, rec, _ in ticks if not obs.ball.valid)
    x_lo, x_hi, y_lo, y_hi = limits(rig)
    prev, last_t = center(rig), ticks[0][0]
    for t, _, rec, _ in ticks:
        assert rec.armed
        p = (rec.servo_cmd.x_us, rec.servo_cmd.y_us)
        assert x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi
        step = rig["slew_us_s"] * (t - last_t) + 1
        assert abs(p[0] - prev[0]) <= step and abs(p[1] - prev[1]) <= step
        prev, last_t = p, t


def test_empty_maze_holds_the_plate_level(maze):
    cfg, ticks = maze
    empty = [(rec, snap) for t, _, rec, snap in ticks if t >= EMPTY_FROM_S]
    assert empty
    assert all(rec.plate_cmd == LEVEL and snap["ball_px"] is None for rec, snap in empty)
    cx, cy = center(cfg["rig"])
    off = [(abs(r.servo_cmd.x_us - cx), abs(r.servo_cmd.y_us - cy)) for r, _ in empty]
    # only ever heads for centre, and is there by the end
    assert all(a[0] >= b[0] and a[1] >= b[1] for a, b in pairwise(off))
    assert off[-1] == (0, 0)


def test_resting_ball_gets_tilted_back_toward_centre(maze):
    _, ticks = maze
    # +tilt_x rolls the ball toward +x, so a ball off centre wants the opposite tilt
    rest = [rec for t, _, rec, _ in ticks if t < REST_UNTIL_S and rec.ball.valid]
    assert rest
    for rec in rest:
        b, cmd = rec.ball, rec.plate_cmd
        for pos, tilt in ((b.x_mm, cmd.tilt_x_deg), (b.y_mm, cmd.tilt_y_deg)):
            assert abs(pos) <= 5 or pos * tilt < 0
