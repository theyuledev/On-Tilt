import time

import pytest
import serial

from src.app import Loop
from src.control.pulse_map import center
from src.io.serial_link import BoardLink, SerialActuator
from src.io.settings import load
from src.sim.plate import SimActuator
from tests.contract.fake_board import FakeBoard

pytestmark = pytest.mark.contract


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


@pytest.fixture
def board():
    return FakeBoard()


@pytest.fixture
def loop(board):
    cfg = load()
    return Loop(cfg, actuator=SerialActuator(BoardLink(board), cfg["rig"]))


def ticks(loop, clock, n, key=None, dt=0.01):
    for _ in range(n):
        clock[0] += dt
        if key:
            loop.keys.set(*key)
        loop.tick(clock[0], dt)


def unplug(board):
    def gone(data):
        raise serial.SerialException("device disconnected")

    board.write = gone


def go_silent(board):
    board.silent = True


def garble(board):
    board.readline = lambda: b"\xff\xfeOK P x\n"


def test_default_actuator_is_sim():
    loop = Loop(load())
    assert isinstance(loop.actuator, SimActuator)
    assert loop.snapshot()["board_lost"] is False


def test_arm_loads_limits_arms_then_pulses_every_tick(loop, board, clock):
    ticks(loop, clock, 3)
    assert board.sent == []
    loop.arm()
    ticks(loop, clock, 3)
    assert [s[0] for s in board.sent] == ["L", "A", "P", "P", "P"]
    assert board.armed


def test_kill_ramps_to_level_then_disarms(loop, board, clock):
    cx, cy = (round(v) for v in center(loop.cfg["rig"]))
    loop.arm()
    ticks(loop, clock, 100, key=(1.0, 0.0))
    assert board.pulses[0] > cx

    n = len(board.sent)
    loop.kill()
    ticks(loop, clock, 300, key=(1.0, 0.0))
    after = board.sent[n:]
    assert after[-1] == "D" and after[-5:-1] == [f"P {cx} {cy}"] * 4
    xs = [int(s.split()[1]) for s in after[:-1]]
    assert len(xs) > 1 and xs == sorted(xs, reverse=True)
    assert not board.armed and not loop.last.armed


def test_no_sim_ball_on_the_real_board(loop, clock):
    loop.arm()
    ticks(loop, clock, 50, key=(1.0, 0.0))
    assert loop.ball is None and not loop.last.ball.valid


def test_board_error_disarms_and_lets_the_page_arm_again(loop, board, clock):
    loop.arm()
    ticks(loop, clock, 3)
    board.armed = False  # board stalled 250 ms, self levelled and dropped out
    ticks(loop, clock, 1)
    assert board.sent[-1].startswith("P")
    snap = loop.snapshot()
    assert not snap["armed"] and not snap["board_lost"] and not loop.safety.armed

    n = len(board.sent)
    ticks(loop, clock, 3)
    assert board.sent[n:] == []
    loop.arm()
    ticks(loop, clock, 3)
    assert [s[0] for s in board.sent[n:]] == ["L", "A", "P", "P", "P"]
    assert board.armed and loop.snapshot()["armed"]


def test_silent_board_disarms_instead_of_latching_lost(loop, board, clock):
    loop.arm()
    ticks(loop, clock, 3)
    go_silent(board)
    ticks(loop, clock, 1)
    snap = loop.snapshot()
    assert not snap["armed"] and not snap["board_lost"]


def test_late_reply_does_not_lock_out_the_next_arm(loop, board, clock):
    loop.arm()
    ticks(loop, clock, 3)
    board.late = True
    ticks(loop, clock, 1)
    assert not loop.snapshot()["armed"] and board._replies  # the P reply landed after the timeout

    board.armed = False  # 250 ms of silence, board self levelled
    n = len(board.sent)
    loop.arm()
    ticks(loop, clock, 3)
    assert [s[0] for s in board.sent[n:]] == ["L", "A", "P", "P", "P"]
    assert board.armed and loop.snapshot()["armed"]


def test_unplugged_at_arm_latches_board_lost_instead_of_raising(loop, board, clock):
    board.unplugged = True
    loop.arm()
    ticks(loop, clock, 2)
    snap = loop.snapshot()
    assert snap["board_lost"] and not snap["armed"]
    assert board.sent == []


@pytest.mark.parametrize("fail", [unplug, garble])
def test_dead_link_marks_board_lost_and_stops_sending(loop, board, clock, fail):
    loop.arm()
    ticks(loop, clock, 3)
    fail(board)
    ticks(loop, clock, 1)
    n = len(board.sent)

    ticks(loop, clock, 5)
    loop.arm()
    ticks(loop, clock, 5)
    assert len(board.sent) == n
    snap = loop.snapshot()
    assert snap["board_lost"] and not snap["armed"]
