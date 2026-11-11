import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import Intent
from src.inputs.prediction import HandPredictor

pytestmark = pytest.mark.unit


def sample(x, t, y=0.0, valid=True):
    return Intent(x, y, valid, t)


def test_a_smooth_ramp_is_closer_to_the_publication_time_position():
    predictor = HandPredictor()
    outputs = [predictor.update(sample(0.2 * t, t), t + 0.04) for t in (0.0, 0.1, 0.2)]
    expected = 0.2 * 0.24
    assert abs(outputs[-1].x - expected) < abs(0.2 * 0.2 - expected)
    assert outputs[-1].x == pytest.approx(expected)
    assert outputs[-1].t_s == 0.2


@pytest.mark.parametrize("xs", [[0.3] * 8, [0.03, -0.03] * 4, [0, 0, 0.8, 0, 0]])
def test_static_alternating_jitter_and_an_isolated_spike_are_not_extrapolated(xs):
    predictor = HandPredictor()
    for i, x in enumerate(xs):
        assert predictor.update(sample(x, i * 0.1), i * 0.1 + 0.05).x == pytest.approx(x)


def test_prediction_uses_the_slower_of_two_consistent_velocities():
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0), 0.05)
    predictor.update(sample(0.05, 0.1), 0.15)
    out = predictor.update(sample(0.5, 0.2), 0.25)
    assert out.x == pytest.approx(0.525)


def test_a_reversal_stops_extrapolating_on_that_axis():
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0, 0.0), 0.04)
    predictor.update(sample(0.1, 0.1, 0.1), 0.14)
    out = predictor.update(sample(0.2, 0.2, 0.05), 0.24)
    assert out.x == pytest.approx(0.24)
    assert out.y == 0.05


@pytest.mark.parametrize("xy", [(1.0, 1.0), (-1.0, 1.0), (3.0, -4.0), (0.0, 0.0)])
def test_every_output_is_finite_and_inside_the_unit_disk(xy):
    predictor = HandPredictor()
    for i in range(5):
        out = predictor.update(sample(xy[0] * i, i * 0.1, xy[1] * i), i * 0.1 + 1.0)
        assert math.isfinite(out.x) and math.isfinite(out.y)
        assert math.hypot(out.x, out.y) <= 1.0 + 1e-12


def test_old_samples_get_at_most_one_observed_frame_of_prediction_and_keep_their_stamp():
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0), 0.05)
    predictor.update(sample(0.1, 0.1), 0.15)
    out = predictor.update(sample(0.2, 0.2), 100.0)
    assert out.x == pytest.approx(0.3)
    assert out.t_s == 0.2


@pytest.mark.parametrize("bad", [sample(0.0, 0.2, valid=False), sample(math.nan, 0.2)])
def test_lost_or_nonfinite_input_is_invalid_and_clears_prediction(bad):
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0), 0.05)
    predictor.update(sample(0.1, 0.1), 0.15)
    out = predictor.update(bad, 0.25)
    assert out == sample(0.0, bad.t_s, valid=False)
    assert not predictor.history
    for x, t in ((0.8, 0.3), (0.9, 0.4)):
        assert predictor.update(sample(x, t), t + 0.05).x == x


@pytest.mark.parametrize("stamp", [0.1, 0.05])
def test_duplicate_or_older_measurements_reset_prediction(stamp):
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0), 0.05)
    predictor.update(sample(0.1, 0.1), 0.15)
    assert predictor.update(sample(0.2, stamp), 0.2) == sample(0.2, stamp)
    assert not predictor.history


def test_recentre_drops_the_old_velocity():
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0), 0.05)
    predictor.update(sample(0.1, 0.1), 0.15)
    assert predictor.update(sample(0.0, 0.2), 0.25, recentre=True) == sample(0.0, 0.2)
    assert predictor.update(sample(0.1, 0.3), 0.35) == sample(0.1, 0.3)


def test_a_single_landmark_glitch_cannot_kick_the_smoothed_prediction():
    from src.inputs.hand import HandIntent
    from src.io.settings import BASE, load

    cfg = load(local=BASE)["input"]
    hand, predictor = HandIntent(cfg), HandPredictor()
    hand.update((0.0, 0.0), 0.0, recentre=True)
    for i in range(1, 60):
        reading = (80.0, -80.0) if i == 30 else (0.0, 0.0)
        t = i / 30
        out = predictor.update(hand.update(reading, t), t + 0.025)
        assert out.valid and (out.x, out.y) == (0.0, 0.0)


def test_extremely_close_timestamps_cannot_produce_nonfinite_prediction():
    predictor = HandPredictor()
    for i in range(3):
        out = predictor.update(sample(0.1 * i, i * 1e-320), 1.0)
        assert math.isfinite(out.x) and math.isfinite(out.y)


@pytest.mark.parametrize("stamp,published", [(math.nan, 1.0), (1.0, math.inf)])
def test_nonfinite_times_are_invalid_and_reset_prediction(stamp, published):
    predictor = HandPredictor()
    predictor.update(sample(0.0, 0.0), 0.05)
    out = predictor.update(sample(0.1, stamp), published)
    assert not out.valid and (out.x, out.y) == (0.0, 0.0)
    assert math.isfinite(out.t_s) and not predictor.history


def test_run_uses_capture_times_and_predicts_with_age_after_inference():
    from src.inputs.hand_process import run

    events, sent, readings = [], [], []
    frames = [SimpleNamespace(image=np.zeros((2, 2)), t_s=t) for t in (1.0, 1.1, 1.2)]
    times = iter((1.04, 1.14, 1.24))

    class Frames:
        @property
        def running(self):
            return bool(frames)

        def read(self):
            return frames.pop(0)

    class Hand:
        seen, calibrating, progress, mode = True, False, 1.0, "tilt"

        def update(self, reading, t, recentre):
            readings.append(t)
            return sample(reading, t)

    def detect(image, t):
        events.append(("detect", t))
        return t - 1.0

    def clock():
        assert events[-1][0] == "detect"
        events.append(("publish", None))
        return next(times)

    run(Frames(), detect, Hand(), lambda intent, **flags: sent.append((intent, flags)), clock=clock)
    assert readings == [1.0, 1.1, 1.2]
    assert sent[-1][0].x == pytest.approx(0.24)
    assert [intent.t_s for intent, _ in sent] == readings
    assert all(flags["lag_s"] == pytest.approx(0.04) for _, flags in sent)
