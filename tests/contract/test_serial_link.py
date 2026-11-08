import pytest

from src.contracts import ServoCommand
from src.io.serial_link import BoardLink, LinkError, SerialActuator, limits_line
from src.io.settings import load
from tests.contract.fake_board import FakeBoard

pytestmark = pytest.mark.contract


@pytest.fixture
def rig():
    # the exact lines below take level as 1500
    return {**load()["rig"], "trim_x_us": 0, "trim_y_us": 0}


@pytest.fixture
def board():
    return FakeBoard()


def test_first_command_loads_limits_arms_then_sends_pulses(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    act.send(ServoCommand(1520, 1480), 0.0)
    assert [s[0] for s in board.sent] == ["L", "A", "P"]
    assert board.sent[0] == "L 1350 1650 1350 1650 1500 1500"
    assert board.armed and board.pulses == (1520, 1480)


def test_board_clamp_is_visible_on_the_raw_link(board, rig):
    link = BoardLink(board)
    link.ask(limits_line(rig))
    link.ask("A")
    assert link.ask("P 1900 1100") == ["1650", "1350"]
    assert board.pulses == (1650, 1350)


def test_actuator_refuses_a_clamped_echo(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    with pytest.raises(LinkError, match="'P 1900 1100' -> 'OK P 1650 1350' echo mismatch"):
        act.send(ServoCommand(1900, 1100), 0.0)
    assert [s[0] for s in board.sent] == ["L", "A", "P"]


def test_stale_p_echo_is_skipped_not_taken_as_this_reply(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    act.send(ServoCommand(1490, 1490), 0.0)
    board._replies.append("OK P 1490 1490")  # last tick's reply turning up late
    state = act.send(ServoCommand(1500, 1500), 0.01)
    assert (state.sent_x_deg, state.sent_y_deg) == (0.0, 0.0)
    assert board.sent[-1] == "P 1500 1500" and board._replies == []


def test_three_stale_p_echoes_fail_as_an_echo_mismatch(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    act.send(ServoCommand(1500, 1500), 0.0)
    board._replies += ["OK P 1490 1490"] * 3
    with pytest.raises(LinkError, match="'P 1500 1500' -> 'OK P 1490 1490' echo mismatch"):
        act.send(ServoCommand(1500, 1500), 0.01)
    assert board.sent[-1] == "P 1500 1500"


def test_none_disarms_once(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    act.send(ServoCommand(1500, 1500), 0.0)
    act.send(None, 0.01)
    act.send(None, 0.02)
    assert not board.armed
    assert board.sent.count("D") == 1


def test_missing_reply_times_out(board, rig):
    board.silent = True
    with pytest.raises(LinkError, match="no reply"):
        SerialActuator(BoardLink(board), rig).send(ServoCommand(1500, 1500), 0.0)


def test_board_error_is_raised(board):
    link = BoardLink(board)
    with pytest.raises(LinkError, match="nolimits"):
        link.ask("A")
    with pytest.raises(LinkError, match="envelope"):
        link.ask("L 900 1650 1350 1650 1500 1500")
    with pytest.raises(LinkError, match="disarmed"):
        link.ask("P 1500 1500")


def test_ask_skips_two_stragglers_but_not_three(board):
    link = BoardLink(board)
    board._replies += ["OK P 1500 1500"] * 2
    assert link.status()["fw"] == "fake"
    board._replies += ["OK P 1500 1500"] * 3
    with pytest.raises(LinkError, match="'S' -> 'OK P"):
        link.status()


def test_ask_never_skips_an_err(board):
    link = BoardLink(board)
    board._replies.append("ERR P disarmed")
    with pytest.raises(LinkError, match="disarmed"):
        link.status()


def test_status(board, rig):
    link = BoardLink(board)
    assert link.status() == {"armed": False, "limits_loaded": False, "since_ms": 0, "fw": "fake"}
    SerialActuator(link, rig).send(ServoCommand(1500, 1500), 0.0)
    assert link.status()["armed"]
