import copy
import math
from itertools import groupby

import pytest
from fastapi.testclient import TestClient

from src.contracts import Intent
from src.games import maze, scoring
from src.games.progress import in_ring
from src.games.session import Game
from src.io.settings import load
from src.loop import Loop
from src.web.server import make_app
from tests.unit.test_web import RACE_FIELDS

pytestmark = pytest.mark.sim

DT = 0.01
KP = 0.01  # intent per mm off the goal
KD = 0.006  # intent per mm/s, brakes the roll
STAGE_S = 10.0  # time to roll into the start ring, staging has no clock
RACING = ("staging", "countdown", "running")


def on_course(game):
    m, r = game.maze, game.race
    if r.state == "running":
        reached = r.progress.reached if r.progress else 0
        return m.points[m.checkpoints[reached]]
    return m.start


class Player:
    seen, calibrating, progress = True, False, 0.0

    def __init__(self, game):
        self.game = game
        self.goal = on_course
        self.loop = None
        self.now = 1000.0

    def latest(self):
        b = self.loop.ball
        if b is None:
            return Intent(0.0, 0.0, True, self.now)
        gx, gy = self.goal(self.game)

        def axis(err, v):
            return max(-1.0, min(1.0, KP * err - KD * v))

        return Intent(axis(gx - b.x_mm, b.vx_mm_s), axis(gy - b.y_mm, b.vy_mm_s), True, self.now)

    def recentre(self):
        pass


def build(cfg):
    game = Game(cfg, maze.load("square"), scoring.step)
    player = Player(game)
    loop = Loop(cfg, hand=player, game=game)
    loop.bus.subscribe("tick", lambda rec: game.tick(rec, loop.safety.killing, loop.board_lost))
    player.loop = loop
    return loop, game, player


def tick(loop, player, n=1):
    for _ in range(n):
        player.now += DT
        loop.tick(player.now, DT)


def send(loop, msg):
    with TestClient(make_app(loop)).websocket_connect("/ws") as sock:
        sock.send_json(msg)
        return sock.receive_json()


def run(loop, player, until, seconds):
    seen = []
    for _ in range(round(seconds / DT)):
        tick(loop, player)
        snap = loop.snapshot()
        seen.append((snap["race"]["state"], snap["mode"]))
        if until(loop.game):
            break
    return seen


def ends(game):
    return game.race.state in ("finished", "dnf", "idle")


def budget(cfg):
    return STAGE_S + cfg["game"]["countdown_s"] + cfg["game"]["max_race_s"]


def started(cfg=None):
    loop, game, player = build(cfg or copy.deepcopy(load()))
    loop.arm()
    tick(loop, player)
    seen = [(loop.snapshot()["race"]["state"], loop.mode)]
    send(loop, {"race": "start"})
    return loop, game, player, seen


def test_full_race_goes_idle_to_finished_and_hands_the_plate_back():
    cfg = copy.deepcopy(load())
    cfg["control"]["mode"] = "balance"  # not hand, so handing the plate back shows
    loop, game, player, seen = started(cfg)
    seen += run(loop, player, ends, budget(cfg))
    assert [s for s, _ in groupby(s for s, _ in seen)] == [
        "idle",
        "staging",
        "countdown",
        "running",
        "finished",
    ]
    assert {m for s, m in seen if s in RACING} == {"hand"}
    view = loop.snapshot()["race"]
    assert view["reached"] == view["score"]["checkpoints"] == len(game.maze.checkpoints)
    assert 0 < view["elapsed_s"] < cfg["game"]["max_race_s"]
    tick(loop, player)
    assert loop.mode == cfg["control"]["mode"] and loop.last.armed


def test_the_page_gets_every_race_field_while_running():
    loop, game, player, _ = started()
    run(loop, player, lambda g: g.race.state == "running", budget(loop.cfg))
    view = send(loop, {})["race"]
    assert view["state"] == "running"
    assert set(RACE_FIELDS) <= view.keys()


def test_abort_from_staging_goes_idle_and_the_plate_carries_on():
    loop, game, player, _ = started()
    tick(loop, player, 5)
    ring = loop.cfg["game"]["checkpoint_r_mm"]
    assert game.race.state == "staging"
    assert not in_ring(loop.last.ball, game.maze.start, ring)
    assert send(loop, {"race": "abort"})["race"]["state"] == "idle"
    tick(loop, player, round(1 / DT))
    assert game.race.state == "idle" and game.race.reason is None
    assert loop.safety.armed and not loop.safety.killing and loop.last.armed
    assert loop.mode == loop.cfg["control"]["mode"]


def test_kill_mid_race_is_a_dnf_and_the_plate_ramps_level_and_disarms():
    loop, game, player, _ = started()
    run(loop, player, lambda g: g.view()["reached"] >= 1, budget(loop.cfg))
    assert game.race.state == "running"
    send(loop, {"kill": True})
    rig = loop.cfg["rig"]
    tilt = []
    # ramp from the edge of the startup range plus a second to settle
    for _ in range(round((rig["startup_range_us"] / rig["kill_ramp_us_s"] + 1.0) / DT)):
        tick(loop, player)
        s = loop.last.plate_state
        tilt.append(math.hypot(s.sent_x_deg, s.sent_y_deg))
    assert (game.race.state, game.race.reason) == ("dnf", "killed")
    assert tilt[0] > 0 and tilt == sorted(tilt, reverse=True) and tilt[-1] == 0
    assert not loop.safety.armed and not loop.last.armed


def test_leaving_the_start_ring_in_the_countdown_goes_back_to_idle():
    loop, game, player, _ = started()
    run(loop, player, lambda g: g.race.state == "countdown", budget(loop.cfg))
    player.goal = lambda g: (0.0, 0.0)
    seen = run(loop, player, ends, loop.cfg["game"]["countdown_s"])
    assert "running" not in {s for s, _ in seen}
    assert (game.race.state, game.race.reason) == ("idle", "left the start")
    ring = loop.cfg["game"]["checkpoint_r_mm"]
    assert not in_ring(loop.last.ball, game.maze.start, ring)
    assert loop.last.armed
