import itertools
import math
import random

import numpy as np
import pytest

from src.io.ports import Frame
from src.io.serial_link import BoardLink, LinkError
from tests.contract.fake_board import FakeBoard
from tests.unit.test_servo_test import Clock
from tools.led_latency import POLL_S, parse, run
from tools.led_math import level, pick_window, saturation_warning, summarize

pytestmark = pytest.mark.unit

PERIOD = 1 / 120
LED_DELAY = 0.004
STEP = 0.0001
WINDOW = (17, 26, 7, 16)


def spot(value, shape=(32, 32)):
    img = np.zeros(shape, np.float32)
    img[10:13, 20:23] = value
    return img


def test_pick_window_centres_on_the_bright_spot_and_clips():
    assert pick_window(np.zeros((32, 32), np.float32), spot(255)) == WINDOW
    assert pick_window(np.zeros((32, 32, 3), np.float32), spot(255, (32, 32, 3))) == WINDOW
    assert pick_window(np.zeros((32, 32), np.float32), spot(255), half=25) == (0, 32, 0, 32)


def test_pick_window_gives_none_under_the_contrast_floor():
    assert pick_window(np.zeros((32, 32), np.float32), spot(20)) is None


def test_level_on_gray_and_bgr():
    gray = np.zeros((8, 8), np.uint8)
    gray[2:4, 2:4] = 200
    bgr = np.zeros((8, 8, 3), np.uint8)
    bgr[2:4, 2:4] = (100, 200, 0)
    assert level(gray, (2, 4, 2, 4)) == 200
    assert level(bgr, (2, 4, 2, 4)) == 100
    assert level(gray, (0, 8, 0, 8)) == 200 * 4 / 64


def test_summarize_arithmetic():
    out = summarize([0.010, 0.012, 0.014, 0.016], [0.002, 0.004], fps=100)
    assert out["n"] == 4 and out["mean"] == pytest.approx(0.013)
    assert out["median"] == pytest.approx(0.013) and out["p50"] == pytest.approx(0.013)
    assert out["p95"] == pytest.approx((0.014 * 3 + 0.016 * 17) / 20)
    assert (out["min"], out["max"]) == (0.010, 0.016)
    assert out["mean_rtt"] == pytest.approx(0.003)
    assert out["l_mid"] == pytest.approx(0.013 - 0.005 - 0.0015)
    assert out["l_min"] == pytest.approx(0.010 - 0.0015)
    assert out["spread"] == pytest.approx(0.006)
    assert summarize([0.01], [0.002], fps=100)["p95"] == 0.01


def test_parse_name_trials_and_roi():
    assert parse(["a"]) == ("a", 50, None)
    assert parse(["a", "7"]) == ("a", 7, None)
    assert parse(["--roi", "10,20,5,6", "a", "7"]) == ("a", 7, (10, 15, 20, 26))
    assert parse(["a", "--roi", "10,20,5,6"]) == ("a", 50, (10, 15, 20, 26))


class LedBoard(FakeBoard):
    """Lights the led a fixed delay after each M 1; delays run out of an iterator."""

    def __init__(self, clock, delays):
        super().__init__()
        self.clock, self.delays, self.t_lit = clock, iter(delays), math.inf

    def _handle(self, parts):
        if parts == ["M", "1"]:
            self.t_lit = self.clock() + next(self.delays)
        return super()._handle(parts)


def picture(lit):
    img = np.full((32, 32), 20, np.uint8)
    if lit:
        img[10:13, 20:23] = 255
    return img


class LedCam:
    """A frame every 1/120 s on the shared clock, newest wins, spot once the board lit it."""

    def __init__(self, board, clock):
        self.board, self.clock, self.next = board, clock, PERIOD

    def read(self):
        if self.clock.t < self.next:
            return None
        t = self.next + math.floor((self.clock.t - self.next) / PERIOD) * PERIOD
        self.next = t + PERIOD
        return Frame(picture(self.board.led and t >= self.board.t_lit), t)

    def close(self):
        pass


class StaleCam(LedCam):
    """Right after each M 1 hands out a lit frame stamped a second ago, a leftover."""

    def __init__(self, board, clock):
        super().__init__(board, clock)
        self.served = 0

    def read(self):
        if self.board.sent.count("M 1") > self.served:
            self.served += 1
            return Frame(picture(True), self.clock.t - 1.0)
        return super().read()


def rig(delays, cam=LedCam):
    clock = Clock(STEP)
    board = LedBoard(clock, delays)

    def sleep(s):
        clock.t += s

    return board, dict(
        link=BoardLink(board),
        cam=cam(board, clock),
        clock=clock,
        sleep=sleep,
        rand=random.Random(1).uniform,
    )


def in_bounds(lat):
    return all(LED_DELAY <= x < LED_DELAY + PERIOD + POLL_S for x in lat)


def test_run_times_each_trial_within_a_frame_and_only_sends_m():
    board, kw = rig(itertools.repeat(LED_DELAY))
    hits = []
    lat, rtt, misses = run(trials=20, on_hit=hits.append, **kw)
    assert len(lat) == 20 and misses == 0 and in_bounds(lat)
    assert [h.t_lit - h.t_cmd for h in hits] == lat
    assert [h.t_ack - h.t_cmd for h in hits] == rtt
    assert all(h.frames_to_lit >= 1 and h.level_dark < h.level_lit for h in hits)
    assert len({round(x, 5) for x in lat}) > 1
    assert all(r > 0 for r in rtt)
    assert set(board.sent) == {"M 0", "M 1"}


def test_a_lit_frame_from_before_the_command_is_skipped():
    board, kw = rig(itertools.repeat(LED_DELAY), StaleCam)
    lat, rtt, misses = run(trials=10, **kw)
    assert len(lat) == 10 and misses == 0 and kw["cam"].served == 11
    assert in_bounds(lat)


def test_a_given_roi_skips_the_auto_pick():
    board, kw = rig(itertools.repeat(LED_DELAY))
    lat, rtt, misses = run(trials=5, roi=WINDOW, **kw)
    assert len(lat) == 5 and misses == 0 and in_bounds(lat)
    # dead led: no bail out on contrast, the trials run and fail their dark check
    board, kw = rig(itertools.repeat(math.inf))
    assert run(trials=2, roi=WINDOW, **kw) == ([], [], 2)
    assert board.sent == ["M 0", "M 1", "M 0", "M 0"]


def test_trials_without_a_lit_frame_count_as_misses():
    board, kw = rig(itertools.chain([LED_DELAY], itertools.repeat(math.inf)))
    assert run(trials=3, **kw) == ([], [], 3)
    assert board.sent.count("M 1") == 4


def test_led_out_of_view_stops_before_any_trial(capsys):
    board, kw = rig(itertools.repeat(math.inf))
    assert run(trials=3, **kw) == ([], [], 0)
    assert board.sent == ["M 0", "M 1"]
    assert "led not in view" in capsys.readouterr().out


def test_a_silent_board_is_an_error_not_a_miss():
    board, kw = rig(itertools.repeat(LED_DELAY))
    board.silent = True
    with pytest.raises(LinkError):
        run(trials=3, **kw)


E, L = 0.003, 0.012


class ExposedCam:
    """Frames every 1/120 s exposing [s, s+E], stamped L after the exposure midpoint; the led
    window reads the share of the exposure that fell after the board lit it."""

    def __init__(self, board, clock):
        self.board, self.clock, self.k = board, clock, 0

    def read(self):
        newest = math.floor((self.clock.t - E / 2 - L) / PERIOD)
        if newest < self.k:
            return None
        self.k = newest + 1
        s = newest * PERIOD
        lit = min(1.0, max(0.0, (s + E - self.board.t_lit) / E)) if self.board.led else 0.0
        img = np.full((32, 32), 20, np.float32)
        img[10:13, 20:23] = 20 + 235 * lit
        return Frame(img, s + E / 2 + L)

    def close(self):
        pass


def test_l_mid_recovers_the_pipeline_delay_under_a_real_exposure_model():
    # delay 0 puts the led on at the board's turn, halfway through the round trip
    _, kw = rig(itertools.repeat(0.0), ExposedCam)
    lat, rtt, misses = run(trials=400, **kw)
    out = summarize(lat, rtt, fps=120)
    assert misses == 0 and abs(out["l_mid"] - L) < 0.0003
    assert L - E / 2 <= out["l_min"] <= L + E / 2


def test_saturation_warning_starts_at_240_on_the_window_peak():
    img = np.zeros((8, 8), np.float32)
    img[3, 3] = 239.9
    assert saturation_warning(img, (0, 8, 0, 8)) is None
    img[3, 3] = 240
    assert "saturation" in saturation_warning(img, (0, 8, 0, 8))
    assert saturation_warning(img, (4, 8, 4, 8)) is None


def test_run_warns_once_when_the_led_spot_saturates(capsys):
    _, kw = rig(itertools.repeat(LED_DELAY))
    run(trials=3, **kw)
    assert capsys.readouterr().out.count("saturation") == 1


class DimCam(LedCam):
    def read(self):
        f = super().read()
        return f and Frame(np.minimum(f.image, 200).astype(np.uint8), f.t_s)


def test_run_stays_quiet_when_the_led_spot_is_dim(capsys):
    _, kw = rig(itertools.repeat(LED_DELAY), DimCam)
    run(trials=3, **kw)
    assert "saturation" not in capsys.readouterr().out
