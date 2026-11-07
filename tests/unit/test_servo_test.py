import os
import signal

import numpy as np
import pytest

from src.io.ports import Frame
from src.io.serial_link import BoardLink
from src.io.settings import load
from src.vision.plate_pose import QuadTracker
from src.vision.pointer import PairTracker
from tests.contract.fake_board import FakeBoard
from tests.unit.test_plate_pose import SIZE, D, K, pose, render
from tests.unit.test_pointer import frame
from tools.bench import Bench
from tools.servo_test import guarded, load_csv, run, save

pytestmark = pytest.mark.unit
TICK = 0.02


class Clock:
    def __init__(self, step=0.004):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class FakeCam:
    def __init__(self, images, clock):
        self.images, self.clock, self.reads = list(images), clock, 0

    def read(self):
        img = self.images[min(self.reads, len(self.images) - 1)]
        self.reads += 1
        return Frame(img, self.clock())

    def close(self):
        pass


@pytest.fixture
def bench():
    board = FakeBoard()
    b = Bench(BoardLink(board), load()["rig"])
    b.submit("l")
    b.submit("A")
    return board, b


def test_run_picks_the_horn_pair_and_logs_both_command_times(bench):
    board, b = bench
    steps = [(1650, "find")] * 3 + [(1500, "find")] * 2 + [(1500, "hold")] * 3
    clock = Clock()
    cam = FakeCam([frame(0)] + [frame(4)] * 40, clock)
    commands, rows = run(b, cam, steps, 1500, 0, TICK, PairTracker(1650), clock)
    assert [s for s in board.sent if s.startswith("P")][:4] == ["P 1650 1500"] * 3 + ["P 1500 1500"]
    assert len(commands) == 8 and all(c[0] < c[1] for c in commands)
    assert commands[0][2:] == (1650, 1500, "find") and commands[-1][4] == "hold"
    assert len(rows) >= 3 and all(a == pytest.approx(4, abs=0.5) for _, a in rows)


def test_run_on_the_y_axis_tracks_the_plate(bench):
    _, b = bench
    r, tx, ty = pose(tip_y=4, tip_x=-2)
    clock = Clock()
    commands, rows = run(
        b,
        FakeCam([render(r)], clock),
        [(1560, "hold")] * 3,
        1500,
        1,
        TICK,
        QuadTracker(K, D, SIZE),
        clock,
    )
    assert all(c[2:4] == (1500, 1560) for c in commands)
    assert len(rows) >= 2 and len(rows[0]) == 11
    assert rows[0][1] == pytest.approx(tx, abs=0.5) and rows[0][2] == pytest.approx(ty, abs=0.5)


def test_run_stops_when_the_board_does_not_answer(bench):
    board, b = bench
    board.silent = True
    with pytest.raises(RuntimeError):
        run(
            b,
            FakeCam([frame(0)], Clock()),
            [(1550, "find")],
            1500,
            0,
            TICK,
            PairTracker(1550),
            Clock(),
        )


def pulses(board):
    return [int(s.split()[1]) for s in board.sent if s.startswith("P")]


def test_guarded_ramps_to_level_and_disarms_on_any_error(bench):
    board, b = bench
    b.submit("P 1530 1500")

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        guarded(b, 0.01, boom)
    xs = pulses(board)
    assert xs == sorted(xs, reverse=True) and board.sent[-2:] == ["P 1500 1500", "D"]
    assert not board.armed and b.last_p is None


def test_guarded_returns_none_on_ctrl_c_and_disarms_after_a_clean_run(bench):
    board, b = bench
    b.submit("P 1530 1500")

    def ctrl_c():
        raise KeyboardInterrupt

    assert guarded(b, 0.01, ctrl_c) is None
    assert board.sent[-2:] == ["P 1500 1500", "D"]
    b.submit("A")
    assert guarded(b, 0.01, lambda: 7) == 7 and board.sent[-1] == "D"


class TwitchyBoard(FakeBoard):
    """Sends a second ctrl-c partway through the kill ramp."""

    def write(self, data):
        super().write(data)
        if len([s for s in self.sent if s.startswith("P")]) == 4:
            os.kill(os.getpid(), signal.SIGINT)


def test_second_ctrl_c_during_the_ramp_does_nothing():
    board = TwitchyBoard()
    b = Bench(BoardLink(board), load()["rig"])
    b.submit("l")
    b.submit("A")
    b.submit("P 1530 1500")

    def ctrl_c():
        raise KeyboardInterrupt

    assert guarded(b, 0.01, ctrl_c) is None
    assert board.sent[-2:] == ["P 1500 1500", "D"] and len(pulses(board)) > 6
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_csv_round_trip_and_old_single_time_rows(tmp_path, monkeypatch):
    import tools.servo_test as st

    monkeypatch.setattr(st, "OUT", tmp_path)
    commands = [(1.0, 1.002, 1600, 1500, "find"), (1.01, 1.012, 1500, 1500, "hold")]
    rows = [(1.005, 2.5, -1.0, *np.arange(8.0))]
    assert load_csv(save("t", commands, rows, "quad")) == (commands, rows, "quad")
    old = tmp_path / "old.csv"
    old.write_text("cmd,1.0,1600,1500,find\nang,1.005,12.5\n")
    assert load_csv(old) == ([(1.0, 1.0, 1600, 1500, "find")], [(1.005, 12.5)], "ang")
