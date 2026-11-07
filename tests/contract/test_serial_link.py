import pytest

from src.contracts import ServoCommand
from src.io.serial_link import BoardLink, LinkError, SerialActuator
from src.io.settings import load
from tests.contract.fake_board import FakeBoard

pytestmark = pytest.mark.contract


@pytest.fixture
def rig():
    return load()["rig"]


@pytest.fixture
def board():
    return FakeBoard()


def test_first_command_loads_limits_arms_then_sends_pulses(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    act.send(ServoCommand(1520, 1480), 0.0)
    assert [s[0] for s in board.sent] == ["L", "A", "P"]
    assert board.sent[0] == "L 1350 1650 1350 1650 1500 1500"
    assert board.armed and board.pulses == (1520, 1480)


def test_board_clamp_shows_up_in_plate_state(board, rig):
    act = SerialActuator(BoardLink(board), rig)
    state = act.send(ServoCommand(1900, 1100), 0.0)
    assert board.pulses == (1650, 1350)
    assert state.sent_x_deg == pytest.approx(150 / rig["us_per_deg_x"])
    assert state.sent_y_deg == pytest.approx(-150 / rig["us_per_deg_y"])


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


def test_status(board, rig):
    link = BoardLink(board)
    assert link.status() == {"armed": False, "limits_loaded": False, "since_ms": 0, "fw": "fake"}
    SerialActuator(link, rig).send(ServoCommand(1500, 1500), 0.0)
    assert link.status()["armed"]
