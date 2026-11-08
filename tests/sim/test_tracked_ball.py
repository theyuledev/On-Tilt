import numpy as np
import pytest

from src.app import Loop
from src.contracts import PlateCommand
from src.io.ports import Frame
from src.io.settings import load
from src.vision.tracker import BallPx

pytestmark = pytest.mark.sim


class FakeFrames:
    def __init__(self):
        self.queue = []

    def read(self):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        pass


class FakeTracker:
    """Reads the ball's x from the frame's only pixel."""

    def __init__(self):
        self.calls = 0
        self.valid = True

    def update(self, image, t_s):
        self.calls += 1
        return BallPx(float(image[0, 0]), 400.0, 40.0, self.valid, t_s)


def frame(x_px, t_s):
    return Frame(np.full((1, 1), x_px, float), t_s)


@pytest.fixture
def rig():
    frames, tracker = FakeFrames(), FakeTracker()
    loop = Loop(load(), frames=frames, tracker=tracker)
    loop.mode = "balance"
    return loop, frames, tracker


def test_fresh_frame_gives_a_valid_ball_in_plate_mm_and_its_pixels_for_the_page(rig):
    loop, frames, tracker = rig
    frames.queue.append(frame(740.0, 1.0))
    rec = loop.tick(1.0, 0.01)
    assert rec.ball.valid and rec.ball.t_s == 1.0
    assert rec.ball.x_mm == pytest.approx(100 / loop.cfg["vision"]["px_per_mm"])
    assert rec.ball.y_mm == 0
    assert rec.plate_cmd.tilt_x_deg < 0
    assert loop.snapshot()["ball_px"] == [740.0, 400.0, 40.0]
    assert loop.frame.t_s == 1.0


def test_no_new_frame_skips_the_tracker_and_an_old_ball_goes_invalid(rig):
    loop, frames, tracker = rig
    frames.queue.append(frame(740.0, 1.0))
    loop.tick(1.0, 0.01)
    stale = loop.cfg["rig"]["stale_intent_s"]
    rec = loop.tick(1.0 + stale, 0.01)
    assert tracker.calls == 1 and rec.ball.valid
    rec = loop.tick(1.0 + stale + 0.01, 0.01)
    assert not rec.ball.valid and rec.plate_cmd == PlateCommand(0.0, 0.0)
    frames.queue.append(frame(740.0, 1.0 + stale + 0.02))
    assert loop.tick(1.0 + stale + 0.02, 0.01).ball.valid


def test_lost_lock_is_an_invalid_ball_with_no_overlay(rig):
    loop, frames, tracker = rig
    frames.queue.append(frame(740.0, 1.0))
    loop.tick(1.0, 0.01)
    tracker.valid = False
    frames.queue.append(frame(0.0, 1.01))
    rec = loop.tick(1.01, 0.01)
    assert not rec.ball.valid and rec.plate_cmd == PlateCommand(0.0, 0.0)
    assert loop.snapshot()["ball_px"] is None


def test_camera_without_a_ball_yet_is_invalid_not_the_sim_ball(rig):
    loop, frames, tracker = rig
    rec = loop.tick(1.0, 0.01)
    assert not rec.ball.valid and loop.ball is None
    assert loop.snapshot()["ball_px"] is None
