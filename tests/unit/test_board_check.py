import pytest

from src.io.serial_link import BoardLink
from src.io.settings import load
from tests.contract.fake_board import FakeBoard
from tools.board_check import check, send, show

pytestmark = pytest.mark.unit


class TimedBoard(FakeBoard):
    def wait(self, s: float) -> None:
        if self.armed and s > 0.25:
            self.armed = False
            self.pulses = None


class LaxBoard(TimedBoard):
    def _handle(self, parts):
        if parts[0] == "L" and not self.armed:
            self.limits = [int(a) for a in parts[1:]]
            return "OK L"
        return super()._handle(parts)


def run(board, answer="y"):
    rig = load()["rig"]
    return check(BoardLink(board), rig, ask=lambda _: answer, sleep=board.wait), rig


def test_good_board_passes_every_step(capsys):
    board = TimedBoard()
    results, _ = run(board)
    assert all(ok for _, ok, _ in results)
    assert show(results)
    assert capsys.readouterr().out.splitlines() == [
        "ok  boots disarmed, fw fake",
        "ok  P while disarmed",
        "ok  A before L",
        "ok  L outside envelope",
        "ok  L inside startup range",
        "ok  arm",
        "ok  P past the limit comes back clamped",
        "ok  P at level",
        "ok  timeout leaves it disarmed",
        "ok  re-arm after the timeout",
        "ok  disarm",
        "ok  disarmed after D",
        "ok  marker led on",
        "ok  marker led off",
    ]
    assert not board.armed and not board.led


def test_sent_lines_and_every_p_inside_startup_range():
    board = TimedBoard()
    _, rig = run(board)
    assert board.sent == [
        "S",
        "P 1500 1500",
        "A",
        "L 1049 1575 1425 1575 1500 1500",
        "L 1425 1575 1425 1575 1500 1500",
        "A",
        "P 1650 1500",
        "P 1500 1500",
        "S",
        "A",
        "D",
        "S",
        "M 1",
        "M 0",
    ]
    r = rig["startup_range_us"]
    for line in board.sent:
        if line.startswith("P"):
            x, y = (int(v) for v in line.split()[1:])
            assert abs(x - 1500) <= r and abs(y - 1500) <= r


@pytest.mark.parametrize("answer", ["n", "", " N ", "yes"])
def test_anything_but_y_at_the_gate_sends_only_s(answer, capsys):
    board = TimedBoard()
    results, _ = run(board, answer)
    assert board.sent == ["S"]
    assert results[-1] == ("servo power off", False, "aborted, nothing sent")
    assert not show(results)
    assert capsys.readouterr().out.splitlines()[-1] == "FAIL servo power off: aborted, nothing sent"


def test_board_taking_an_out_of_envelope_l_fails_and_ends_disarmed():
    board = LaxBoard()
    results, _ = run(board)
    assert results[-1][:2] == ("L outside envelope", False)
    assert board.sent[-2:] == ["L 1049 1575 1425 1575 1500 1500", "D"]
    assert not board.armed


def test_limits_already_loaded_skips_a_before_l(capsys):
    board = TimedBoard()
    board.limits = [1350, 1650, 1350, 1650, 1500, 1500]
    results, _ = run(board)
    assert results[2][:2] == ("A before L", None)
    assert board.sent[:3] == ["S", "P 1500 1500", "L 1049 1575 1425 1575 1500 1500"]
    assert show(results)
    assert "skip A before L: limits already loaded" in capsys.readouterr().out


def test_armed_at_start_fails_and_sends_nothing_else():
    board = TimedBoard()
    board.limits = [1350, 1650, 1350, 1650, 1500, 1500]
    board.armed = True
    results, _ = run(board)
    assert board.sent == ["S"]
    assert [ok for _, ok, _ in results] == [False]


def test_board_that_never_times_out_fails_there_and_gets_a_d():
    board = TimedBoard()
    board.wait = lambda s: None
    results, _ = run(board)
    assert results[-1][:2] == ("timeout leaves it disarmed", False)
    assert board.sent[-1] == "D" and not board.armed


def test_silent_board_mid_run_gets_a_d():
    board = TimedBoard()
    board.wait = lambda s: setattr(board, "silent", True)
    results, _ = run(board)
    assert results[-1][:2] == ("timeout leaves it disarmed", False)
    assert board.sent[-1] == "D"


def test_p_past_startup_range_is_never_sent():
    board = TimedBoard()
    rig = {**load()["rig"], "startup_range_us": 150}
    with pytest.raises(ValueError):
        send(BoardLink(board), rig, "P 1651 1500")
    assert board.sent == []
