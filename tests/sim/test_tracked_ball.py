import math

import numpy as np
import pytest

from src.app import Loop
from src.contracts import BallState, PlateCommand
from src.io.ports import Frame
from src.io.settings import load
from src.vision.pipeline import Observation
from src.vision.tracker import BallPx

pytestmark = pytest.mark.sim


class FakeFrames:
    """A camera whose pipeline result is whatever the test last set."""

    def __init__(self):
        self.queue = []
        self.obs = None

    def read(self):
        return self.queue.pop(0) if self.queue else None

    def latest(self):
        return self.obs

    def close(self):
        pass


def see(frames, x_mm, t_s, valid=True):
    frames.queue.append(Frame(np.zeros((1, 1)), t_s))
    px = BallPx(740.0, 400.0, 40.0, valid, t_s)
    ball = BallState(x_mm, 0.0, 0.0, 0.0, valid, t_s)
    frames.obs = Observation(ball, px, math.nan, math.nan, False, t_s, 1.0)


@pytest.fixture
def rig():
    frames = FakeFrames()
    loop = Loop(load(), frames=frames)
    loop.mode = "balance"
    return loop, frames


def test_fresh_frame_gives_a_valid_ball_in_plate_mm_and_its_pixels_for_the_page(rig):
    loop, frames = rig
    see(frames, 50.0, 1.0)
    rec = loop.tick(1.0, 0.01)
    assert rec.ball.valid and rec.ball.t_s == 1.0 and rec.ball.x_mm == 50.0
    assert rec.plate_cmd.tilt_x_deg < 0
    assert loop.snapshot()["ball_px"] == [740.0, 400.0, 40.0]
    assert loop.frame.t_s == 1.0


def test_no_new_result_keeps_the_ball_and_an_old_ball_goes_invalid(rig):
    loop, frames = rig
    see(frames, 50.0, 1.0)
    loop.tick(1.0, 0.01)
    stale = loop.cfg["rig"]["stale_intent_s"]
    assert loop.tick(1.0 + stale, 0.01).ball.valid
    rec = loop.tick(1.0 + stale + 0.01, 0.01)
    assert not rec.ball.valid and rec.plate_cmd == PlateCommand(0.0, 0.0)
    see(frames, 50.0, 1.0 + stale + 0.02)
    assert loop.tick(1.0 + stale + 0.02, 0.01).ball.valid


def test_lost_lock_is_an_invalid_ball_with_no_overlay(rig):
    loop, frames = rig
    see(frames, 50.0, 1.0)
    loop.tick(1.0, 0.01)
    see(frames, 0.0, 1.01, valid=False)
    rec = loop.tick(1.01, 0.01)
    assert not rec.ball.valid and rec.plate_cmd == PlateCommand(0.0, 0.0)
    assert loop.snapshot()["ball_px"] is None


def test_camera_without_a_ball_yet_is_invalid_not_the_sim_ball(rig):
    loop, frames = rig
    rec = loop.tick(1.0, 0.01)
    assert not rec.ball.valid and loop.ball is None
    assert loop.snapshot()["ball_px"] is None


def test_a_result_behind_the_newest_frame_keeps_its_own_stamp(rig):
    loop, frames = rig
    see(frames, 50.0, 1.0)
    frames.queue.append(Frame(np.zeros((1, 1)), 1.02))
    loop.tick(1.03, 0.01)
    loop.tick(1.03, 0.01)
    assert loop.frame.t_s == 1.02 and loop.ball.t_s == 1.0
