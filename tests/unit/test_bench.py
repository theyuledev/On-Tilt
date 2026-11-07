import pytest

from src.io.serial_link import BoardLink
from src.io.settings import load
from tests.contract.fake_board import FakeBoard
from tools.bench import Bench

pytestmark = pytest.mark.unit


@pytest.fixture
def setup():
    board = FakeBoard()
    rig = load()["rig"]
    return board, Bench(BoardLink(board), rig), rig


def armed_at(bench, x, y):
    bench.submit("l")
    bench.submit("A")
    bench.submit(f"P {x} {y}")


def test_l_sends_startup_limits(setup):
    board, bench, _ = setup
    bench.submit("l")
    assert board.sent == ["L 1350 1650 1350 1650 1500 1500"]
    assert board.limits is not None


def test_board_errors_are_logged_not_raised(setup):
    board, bench, _ = setup
    bench.submit("A")
    bench.submit("P 1500 1500")
    assert "ERR A nolimits" in bench.log[0]
    assert "ERR P disarmed" in bench.log[1]
    assert bench.last_p is None


def test_heartbeat_resends_last_p_until_quiet(setup):
    board, bench, _ = setup
    armed_at(bench, 1600, 1400)
    bench.tick(0.01)
    bench.tick(0.01)
    assert board.sent[-2:] == ["P 1600 1400", "P 1600 1400"]
    bench.submit("quiet")
    n = len(board.sent)
    bench.tick(0.01)
    assert len(board.sent) == n


def test_kill_ramps_to_level_then_disarms(setup):
    board, bench, rig = setup
    armed_at(bench, 1650, 1500)
    bench.kill()
    steps = 0
    while bench.killing and steps < 1000:
        bench.tick(0.01)
        steps += 1
    xs = [int(s.split()[1]) for s in board.sent if s.startswith("P")][1:]
    assert xs == sorted(xs, reverse=True)
    assert max(a - b for a, b in zip(xs, xs[1:], strict=False)) <= rig["kill_ramp_us_s"] * 0.01 + 1
    assert board.sent[-2:] == ["P 1500 1500", "D"]
    assert not board.armed and bench.last_p is None


def test_kill_with_no_pulses_just_disarms(setup):
    board, bench, _ = setup
    bench.kill()
    assert board.sent == ["D"]


def test_second_kill_mid_ramp_does_not_restart_the_ramp(setup):
    board, bench, _ = setup
    armed_at(bench, 1650, 1500)
    bench.kill()
    for _ in range(20):
        bench.tick(0.01)
    mid = int(board.sent[-1].split()[1])
    bench.kill()
    bench.tick(0.01)
    assert int(board.sent[-1].split()[1]) <= mid
