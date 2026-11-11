import time

import pytest

from src.app import Loop
from src.control.pulse_map import center
from src.io.serial_link import BoardLink, SerialActuator
from src.io.settings import load
from src.safety import LEVEL_HOLD_S
from tests.contract.fake_board import FakeBoard
from tests.contract.test_loop_board import go_silent, ticks

pytestmark = pytest.mark.contract

KEY = (1.0, -0.25)  # x at its limit, y part way the other side
DT = 0.01
TICKS = 300


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


def toward(v, target, step):
    if abs(v - target) <= step:
        return target
    return v - step if v > target else v + step


def kill_lines(start, level, rate, dts):
    # each axis walks to level at rate * dt, then the level P repeats for the hold, then D
    # the hold counts from the first level P, so the tick that gets there adds nothing
    (x, y), (cx, cy), held, out = start, level, None, []
    for dt in dts:
        x, y = toward(x, cx, rate * dt), toward(y, cy, rate * dt)
        out.append(f"P {round(x)} {round(y)}")
        if (x, y) == (cx, cy):
            held = 0.0 if held is None else held + dt
            if held >= LEVEL_HOLD_S:
                return [*out, "D"]
    raise AssertionError("ramp never reached level")


def tilt(loop, board, clock):
    loop.arm()
    ticks(loop, clock, 100, key=KEY)
    x0, y0 = board.pulses
    cx, cy = center(loop.cfg["rig"])
    assert board.sent[-1] == f"P {x0} {y0}" and x0 > cx and y0 < cy
    return (x0, y0), len(board.sent)


def plan(loop, start, dts):
    rig = loop.cfg["rig"]
    return kill_lines(start, center(rig), rig["kill_ramp_us_s"], dts)


def kill(loop, clock, dts):
    loop.kill()
    for dt in dts:
        ticks(loop, clock, 1, key=KEY, dt=dt)


def level_line(loop):
    cx, cy = center(loop.cfg["rig"])
    return f"P {round(cx)} {round(cy)}"


SLOW = {
    "even": None,
    "slow_in_ramp": lambda arrive: arrive // 2,
    "slow_on_arrival": lambda arrive: arrive,
    "slow_in_hold": lambda arrive: arrive + 1,
}


@pytest.mark.parametrize("slow", SLOW.values(), ids=SLOW.keys())
def test_kill_from_a_tilt_sends_the_plain_ramp_then_level_then_d(loop, board, clock, slow):
    start, n = tilt(loop, board, clock)
    level = level_line(loop)
    dts = [DT] * TICKS
    if slow:
        dts[slow(plan(loop, start, dts).index(level))] = 0.05
    want = plan(loop, start, dts)

    kill(loop, clock, dts)
    after = board.sent[n:]
    assert after == want
    assert after[-2:] == [level, "D"]
    # D comes a full hold after the first level P, whichever tick was slow
    first = after.index(level)
    assert after.count(level) >= 2
    assert sum(dts[first + 1 : len(after) - 1]) >= LEVEL_HOLD_S
    assert not board.armed and not loop.snapshot()["armed"]


@pytest.mark.parametrize("attr, value", [("armed", False), ("ramping", True)])
def test_err_part_way_down_stops_without_d_or_s(loop, board, clock, attr, value):
    start, n = tilt(loop, board, clock)
    want = plan(loop, start, [DT] * TICKS)
    loop.kill()
    ticks(loop, clock, 20, key=KEY)
    setattr(board, attr, value)  # its own stall timeout already took over
    ticks(loop, clock, TICKS - 20, key=KEY)

    assert board.sent[n:] == want[:21]
    snap = loop.snapshot()
    assert not snap["armed"] and not snap["board_lost"]


def late(board):
    board.late = True


def late_while_ramping(board):
    board.late = board.ramping = True


@pytest.mark.parametrize("fail", [late, late_while_ramping, go_silent])
def test_no_d_after_an_unacked_last_level_pulse(loop, board, clock, fail):
    start, n = tilt(loop, board, clock)
    want = plan(loop, start, [DT] * TICKS)
    last = len(want) - 2  # tick that sends the final level P
    loop.kill()
    ticks(loop, clock, last, key=KEY)
    fail(board)
    ticks(loop, clock, TICKS - last, key=KEY)

    assert board.sent[n:] == [*want[:-1], "S"]
    snap = loop.snapshot()
    assert not snap["armed"] and not snap["board_lost"]
