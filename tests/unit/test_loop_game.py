import pytest

from src.contracts import BallState
from src.games.session import Game
from src.io.settings import load
from src.loop import Loop
from tests.unit.test_loop import MISMATCH, Hand
from tests.unit.test_race import GAME, MAZE, fake_step

pytestmark = pytest.mark.unit


class FakeGame:
    def __init__(self, mode=None, fault_at=None):
        self.m, self.fault_at, self.ticks = mode, fault_at, 0

    def mode(self):
        return self.m

    def view(self):
        return {"state": "staging", "ticks": self.ticks}

    def tick(self, rec, killing, board_lost):
        self.ticks += 1
        if self.ticks == self.fault_at:
            self.m = None


def game_loop(game, exposure=None):
    cfg = load()
    cfg["control"]["mode"] = "balance"
    return Loop(cfg, exposure=exposure, game=game)


def test_the_loop_runs_the_mode_the_game_asks_for():
    game = FakeGame("hand")
    loop = game_loop(game)
    loop.tick(1.0, 0.01)
    assert loop.mode == "hand"
    game.m = None
    loop.tick(1.01, 0.01)
    assert loop.mode == "balance"
    loop = Loop(load())
    loop.mode = "path"
    loop.tick(1.0, 0.01)
    assert loop.mode == "path"


@pytest.mark.parametrize("exposure", [MISMATCH, None])
def test_switching_armed_into_a_blocked_mode_ramps_to_level(exposure):
    game = FakeGame("hand")
    loop = game_loop(game, exposure)
    loop.tick(1.0, 0.01)
    loop.arm()
    assert loop.safety.armed
    game.m = None
    loop.tick(1.01, 0.01)
    assert loop.mode == "balance" and loop.safety.armed
    assert loop.safety.killing is (exposure is not None)


def test_the_snapshot_carries_the_race_view_only_with_a_game():
    game = FakeGame()
    loop = game_loop(game)
    loop.tick(1.0, 0.01)
    assert loop.snapshot()["race"] == game.view()
    assert "race" not in Loop(load()).snapshot()


def test_a_game_that_drops_out_mid_race_levels_the_plate_and_the_loop_carries_on():
    game = FakeGame("hand", fault_at=3)
    loop = game_loop(game, MISMATCH)
    loop.bus.subscribe("tick", lambda rec: game.tick(rec, loop.safety.killing, loop.board_lost))
    loop.tick(1.0, 0.01)
    loop.arm()
    recs = [loop.tick(1.0 + i * 0.01, 0.01) for i in range(1, 6)]
    assert len(recs) == 5 and game.ticks == 6
    assert loop.mode == "balance" and loop.safety.killing
    loop.arm()
    assert loop.arm_blocked == MISMATCH


def real_game(step=fake_step):
    cfg = load()
    cfg["control"]["mode"] = "balance"
    cfg["game"] = {**GAME, "countdown_s": 0.0, "attract": False}
    game, hand = Game(cfg, MAZE, step), Hand()
    loop = Loop(cfg, hand, game=game)
    loop.bus.subscribe("tick", lambda rec: game.tick(rec, loop.safety.killing, loop.board_lost))
    return loop, game, hand


def at(loop, hand, t, x=-100.0):
    hand.t = t
    loop.ball = BallState(x, 0.0, 0.0, 0.0, True, t)
    return loop.tick(t, 0.01)


def test_a_game_mode_bug_falls_back_to_the_settings_mode(capsys):
    loop, game, hand = real_game()
    loop.mode = "hand"
    game.race = object()
    at(loop, hand, 1.0)
    assert loop.mode == "balance" and game.off
    assert capsys.readouterr().out.count("race off") == 1
    recs = [at(loop, hand, 1.0 + i * 0.01) for i in range(1, 4)]
    assert [r.mode for r in recs] == ["balance"] * 3 and loop.snapshot()["race"] == {}


def test_a_scoring_bug_mid_race_ends_the_race_and_the_loop_carries_on(capsys):
    def boom(*a):
        raise ValueError("bad gate")

    loop, game, hand = real_game(boom)
    at(loop, hand, 1.0)
    loop.arm()
    at(loop, hand, 1.01)
    game.start()
    at(loop, hand, 1.02)
    assert game.race.state == "running" and loop.mode == "hand"
    at(loop, hand, 1.03, x=-60.0)
    assert game.off and capsys.readouterr().out.count("race off") == 1
    recs = [at(loop, hand, 1.03 + i * 0.01, x=-60.0) for i in range(1, 4)]
    assert [r.mode for r in recs] == ["balance"] * 3 and all(r.armed for r in recs)
    assert loop.snapshot()["race"] == {}
