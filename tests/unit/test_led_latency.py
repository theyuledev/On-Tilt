import itertools
import math
import random

import numpy as np
import pytest

from src.io.ports import Frame
from src.io.serial_link import BoardLink, LinkError
from tests.contract.fake_board import FakeBoard
from tests.unit.test_servo_test import Clock
from tools.led_latency import level, pick_window, run, summarize

pytestmark = pytest.mark.unit

PERIOD = 1 / 120
LED_DELAY = 0.004
STEP = 0.0001


def spot(value, shape=(32, 32)):
    img = np.zeros(shape, np.float32)
    img[10:13, 20:23] = value
    return img


def test_pick_window_centres_on_the_bright_spot_and_clips():
    assert pick_window(np.zeros((32, 32), np.float32), spot(255)) == (17, 26, 7, 16)
    assert pick_window(np.zeros((32, 32, 3), np.float32), spot(255, (32, 32, 3))) == (17, 26, 7, 16)
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
    out = summarize([0.010, 0.012, 0.014, 0.016], [0.002, 0.004], fps=100, exposure_s=0.002)
    assert out["n"] == 4 and out["mean"] == pytest.approx(0.013)
    assert out["median"] == pytest.approx(0.013)
    assert (out["min"], out["max"]) == (0.010, 0.016)
    assert out["mean_rtt"] == pytest.approx(0.003)
    assert out["l_mid"] == pytest.approx(0.013 - 0.005 - 0.001 - 0.0015)
    assert out["l_min"] == pytest.approx(0.010 - 0.0015)
    assert out["spread"] == pytest.approx(0.006)


class LedBoard(FakeBoard):
    """Lights the led a fixed delay after each M 1; delays run out of an iterator."""

    def __init__(self, clock, delays):
        super().__init__()
        self.clock, self.delays, self.t_lit = clock, iter(delays), math.inf

    def _handle(self, parts):
        if parts == ["M", "1"]:
            self.t_lit = self.clock() + next(self.delays)
        return super()._handle(parts)


class LedCam:
    """A frame every 1/120 s on the shared clock, newest wins, spot once the board lit it."""

    def __init__(self, board, clock):
        self.board, self.clock, self.next = board, clock, PERIOD

    def read(self):
        if self.clock.t < self.next:
            return None
        t = self.next + math.floor((self.clock.t - self.next) / PERIOD) * PERIOD
        self.next = t + PERIOD
        img = np.full((32, 32), 20, np.uint8)
        if self.board.led and t >= self.board.t_lit:
            img[10:13, 20:23] = 255
        return Frame(img, t)

    def close(self):
        pass


def rig(delays):
    clock = Clock(STEP)
    board = LedBoard(clock, delays)

    def sleep(s):
        clock.t += s

    return board, dict(
        link=BoardLink(board),
        cam=LedCam(board, clock),
        fps=120,
        clock=clock,
        sleep=sleep,
        rand=random.Random(1).uniform,
    )


def test_run_times_each_trial_within_a_frame_and_only_sends_m():
    board, kw = rig(itertools.repeat(LED_DELAY))
    rows = []
    lat, rtt, misses = run(trials=20, on_row=lambda a, b: rows.append((a, b)), **kw)
    assert len(lat) == 20 and misses == 0
    assert rows == list(zip(lat, rtt, strict=True))
    # t0 is one clock tick before the board sees M 1, then the first frame at or after t_lit
    assert all(-1e-9 <= x - (LED_DELAY + STEP) <= PERIOD + 1e-9 for x in lat)
    assert len({round(x, 5) for x in lat}) > 1
    assert all(r > 0 for r in rtt)
    assert set(board.sent) == {"M 0", "M 1"}


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
