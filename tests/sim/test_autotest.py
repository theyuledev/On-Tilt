import time

import pytest

from src.contracts import BallState, PlateCommand
from src.io.settings import load
from tests.contract import timed_board
from tests.contract.timed_board import TimedBoard
from tools import autotest, autotest_board

pytestmark = pytest.mark.sim

CHECKS = ["kill", "stale intent", "board timeout"]


def board_run(board=TimedBoard):
    rig = autotest.sim_rig(load(), board)
    return dict(autotest_board.board_checks(rig)), rig.tap.port


def test_sim_passes_every_check_in_well_under_a_second(capsys):
    start = time.perf_counter()
    rig = autotest.sim_rig(load())
    assert autotest.show(autotest.board_checks(rig) + autotest.hold(rig))
    assert time.perf_counter() - start < 1.0
    lines = capsys.readouterr().out.splitlines()
    print("\n".join(lines))
    assert lines[:3] == [f"ok  {c}" for c in CHECKS]
    assert [line.split()[2] for line in lines[3:]] == [
        "t20_s",
        "overshoot_pct",
        "after_5_s_mm",
        "after_15_s_mm",
        "rms_mm",
    ]
    assert all(line.startswith("ok  hold ") for line in lines[3:])
    assert not rig.tap.port.armed


def test_sim_main_exits_clean(monkeypatch):
    monkeypatch.setattr("sys.argv", ["autotest", "--sim"])
    autotest.main()


class KeepsArmedOnD(TimedBoard):
    def _handle(self, parts):
        return "OK D" if parts == ["D"] else super()._handle(parts)


def test_board_that_stays_armed_after_d_fails_kill():
    results, _ = board_run(KeepsArmedOnD)
    assert results["kill"] == "S says armed 1"


class DisarmsOnStale(autotest_board.Loop):
    def tick(self, now, dt):
        rec = super().tick(now, dt)
        if self.stale:
            self.kill()
        return rec


def test_loop_that_disarms_on_stale_fails_only_the_stale_check(monkeypatch):
    monkeypatch.setattr(autotest_board, "Loop", DisarmsOnStale)
    results, board = board_run()
    assert results == {
        "kill": None,
        "stale intent": "sent D on a stale intent",
        "board timeout": None,
    }
    assert not board.armed


def test_board_that_never_times_out_fails_the_timeout_and_ends_disarmed(monkeypatch):
    monkeypatch.setattr(timed_board, "TIMEOUT_S", float("inf"))
    results, board = board_run()
    assert results["board timeout"].startswith("still armed after 0.")
    assert results["kill"] is None and results["stale intent"] is None
    assert board.sent[-1] == "D" and not board.armed


def test_a_bigger_tilt_waits_longer_and_still_passes_the_timeout(monkeypatch):
    assert autotest_board.quiet_s(150) > autotest_board.quiet_s(105)
    # 150 us takes 790 ms to drop out, past a flat 0.7 s wait
    monkeypatch.setattr(autotest_board, "TILT_US", 140)
    results, board = board_run()
    assert results["board timeout"] is None
    assert board.since_ms == round(autotest_board.quiet_s(150) * 1000)


class DropsOneReply(TimedBoard):
    def write(self, data):
        self.late = len(self.sent) == 4
        super().write(data)


def test_a_missing_reply_fails_the_check_it_landed_in():
    results, _ = board_run(DropsOneReply)
    assert results == {"kill": "board said 'nothing'", "stale intent": None, "board timeout": None}


def test_ball_near_the_rim_aborts_the_hold_and_levels():
    loops = []

    def at_rim(rig):
        loops.append(autotest.Loop(rig.cfg))
        loops[0].ball = BallState(105.0, 0.0, 0.0, 0.0, valid=True, t_s=rig.now())
        return loops[0]

    rig = autotest.sim_rig(load())
    rig.hold_loop = at_rim
    assert autotest.hold(rig) == [("balance hold", "ball near the rim at (105, 0) mm")]
    assert not loops[0].safety.armed
    assert loops[0].actuator.sent == PlateCommand(0.0, 0.0)
