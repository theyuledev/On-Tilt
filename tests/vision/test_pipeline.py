import math

import pytest

from src.io.settings import load
from src.vision import pipeline
from src.vision.pipeline import Pipeline
from tests.unit.test_balls import ball, plate

pytestmark = pytest.mark.unit

CFG = load()
VIS = CFG["vision"]
DT = 1 / 120
SPAN = VIS["velocity_span_frames"]


@pytest.fixture(autouse=True)
def no_local_files(monkeypatch, tmp_path):
    # no lens or board file: the ball only path, whatever this mac has in configs
    monkeypatch.setattr(pipeline, "CALIB_PATH", tmp_path / "calib.json")
    monkeypatch.setattr(pipeline, "BOARD_PATH", tmp_path / "board.json")


def rolling(n, start=0):
    # 10 px right and 10 px down the image every frame: 1200 px/s
    return [
        (ball(plate(), 640 + 10 * i, 400 + 10 * i, 40), i * DT) for i in range(start, start + n)
    ]


def run(pipe, frames):
    return [pipe.step(image, t) for image, t in frames]


def test_ball_time_is_the_read_time_and_the_velocity_spans_the_frames():
    obs = run(Pipeline(CFG), rolling(SPAN + 1))
    assert all(o.ball.valid and o.ball.t_s == o.t_read for o in obs)
    assert obs[-1].t_read == SPAN * DT
    speed = 1200 / VIS["px_per_mm"]
    assert obs[-1].ball.vx_mm_s == pytest.approx(speed, rel=0.1)
    assert obs[-1].ball.vy_mm_s == pytest.approx(-speed, rel=0.1)


def test_same_frame_twice_is_the_same_observation():
    pipe = Pipeline(CFG)
    frames = rolling(3)
    first = run(pipe, frames)[-1]
    assert pipe.step(*frames[-1]) is first
    assert pipe.latest() is first


def test_relock_after_a_loss_starts_the_velocity_from_zero():
    pipe = Pipeline(CFG)
    n = SPAN + 1
    run(pipe, rolling(n))
    empty = plate()
    lost = [pipe.step(empty, (n + i) * DT) for i in range(VIS["lost_after_misses"])]
    assert not lost[-1].ball.valid
    back = run(pipe, rolling(1, start=n + VIS["lost_after_misses"]))[0]
    assert back.ball.valid and (back.ball.vx_mm_s, back.ball.vy_mm_s) == (0.0, 0.0)


def test_tilt_is_nan_and_nothing_occluded_without_markers():
    o = run(Pipeline(CFG), rolling(1))[0]
    assert math.isnan(o.tilt_x_deg) and math.isnan(o.tilt_y_deg)
    assert o.occluded is False
    assert o.ms > 0
