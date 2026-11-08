import pytest

from src.io.serial_link import BoardLink
from src.io.settings import load
from tests.contract.fake_board import FakeBoard
from tools.bench import Bench, headroom, offset

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


def test_l_range_loads_limits_around_center_and_logs_headroom(setup):
    board, bench, rig = setup
    bench.submit("l 200")
    assert board.sent == ["L 1300 1700 1300 1700 1500 1500"]
    assert bench.log[-1] == "headroom x: 250 below, 250 above | y: 250 below, 250 above"
    assert (rig["limit_x_us"], rig["limit_y_us"], rig["startup_flag"]) == (200, 200, False)
    bench.submit("l")
    assert board.sent[-1] == "L 1300 1700 1300 1700 1500 1500"


def test_plain_l_logs_headroom_for_settings_limits(setup):
    board, bench, _ = setup
    bench.submit("l")
    assert bench.log[-1] == headroom(load()["rig"])
    assert headroom({**load()["rig"], "trim_x_us": 20, "trim_y_us": -10}) == (
        "headroom x: 320 below, 280 above | y: 290 below, 310 above"
    )


def test_nudge_moves_live_pulses_through_submit(setup):
    board, bench, _ = setup
    bench.nudge(5, 0)
    assert board.sent == []
    armed_at(bench, 1500, 1500)
    bench.nudge(-5, 0)
    bench.nudge(0, 5)
    assert board.sent[-2:] == ["P 1495 1500", "P 1495 1505"]
    assert bench.last_p == (1495, 1505)
    bench.kill()
    bench.nudge(5, 5)
    assert board.sent[-1] == "P 1495 1505"


def test_offset_reads_as_trim_from_center(setup):
    _, _, rig = setup
    assert offset((1510, 1495), rig) == "off (+10, -5) us"
    assert offset((1500, 1500), {**rig, "trim_x_us": 10}) == "off (-10, +0) us"
