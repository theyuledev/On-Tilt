import pytest

from src.games import ghost, session
from src.games.progress import in_ring
from src.games.session import Game
from src.io.settings import load
from tests.unit.test_race import FINISH, GAME, MAZE, RIG, fake_step, rec

pytestmark = pytest.mark.unit

CFG = {"game": {**GAME, "attract": False}, "rig": RIG}
APPROACH = [(0.0, 50.0, 50.0), (1.0, -100.0, 0.0), (2.0, -100.0, 0.0)]
LAP = [(3.0, -60.0, 0.0), *FINISH]
TAIL = [(11.0, 100.0, 100.0), (12.0, 0.0, 0.0)]


def records(path):
    return [rec(t, x, y) for t, x, y in path]


def race_to_finish(g, exit_t=4.0, path=FINISH):
    g.tick(rec(0.0), False, False)
    g.start()
    g.tick(rec(3.0), False, False)
    g.tick(rec(exit_t, -60.0), False, False)
    for t, x, y in path:
        g.tick(rec(t, x, y), False, False)


def test_lap_cuts_approach_and_tail():
    recs = records(APPROACH + LAP + TAIL)
    lap = ghost.lap(recs, MAZE, GAME, RIG, fake_step)
    assert lap == records(LAP)
    assert ghost.lap(records(APPROACH + LAP[:-1] + TAIL[1:]), MAZE, GAME, RIG, fake_step) == []
    assert ghost.lap([], MAZE, GAME, RIG, fake_step) == []


def test_lap_accepts_a_saved_race_that_starts_outside_the_ring():
    assert ghost.lap(records(LAP), MAZE, GAME, RIG, fake_step) == records(LAP)


def test_splits_are_on_the_lap_clock():
    assert ghost.splits(records(LAP), MAZE, GAME, RIG, fake_step) == [2.0, 5.0, 7.0]
    assert ghost.splits([], MAZE, GAME, RIG, fake_step) == []


def test_tick_keeps_records_from_the_ring_exit():
    g = Game(CFG, MAZE, fake_step)
    g.start()  # no record yet
    assert g.race.state == "idle"
    race_to_finish(g)
    assert g.race.state == "finished"
    assert [r.t_s for r in g.records] == [3.0, 4.0, *(t for t, _, _ in FINISH)]
    assert in_ring(g.records[0].ball, MAZE.start, GAME["checkpoint_r_mm"])
    assert g.records[1].t_s == g.race.t0_s
    g.tick(rec(11.0), False, False)
    assert len(g.records) == 7
    g.start()
    assert g.records == [] and g.race.state == "countdown"


def test_saved_race_cuts_at_the_first_exit_even_if_the_ball_comes_back():
    g = Game(CFG, MAZE, fake_step)
    g.tick(rec(0.0), False, False)
    g.start()
    for t, x in [(1.0, -100.0), (3.0, -100.0), (4.0, -60.0), (4.5, -100.0), (4.8, -60.0)]:
        g.tick(rec(t, x), False, False)
    for t, x, y in FINISH:
        g.tick(rec(t, x, y), False, False)
    assert g.race.state == "finished"
    lap = ghost.lap(g.records, MAZE, GAME, RIG, fake_step)
    assert lap == g.records[1:]
    assert lap[0].t_s == g.race.t0_s == 4.0


def test_save_writes_the_run_only_when_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "ROOT", tmp_path)
    g = Game(CFG, MAZE, fake_step)
    g.save("abc")
    assert not (tmp_path / "data").exists()
    race_to_finish(g)
    g.save("abc")
    path = tmp_path / "data" / "runs" / "abc-7000.jsonl"
    assert ghost.load(path) == g.records


def test_save_makes_the_first_or_a_faster_run_the_ghost(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "ROOT", tmp_path)
    g = Game(CFG, MAZE, fake_step)
    race_to_finish(g)
    g.save("LUCAS")
    v = g.view()
    assert g.trace and v["ghost_lap_s"] == v["elapsed_s"] == 6.0
    race_to_finish(g, exit_t=3.5)
    g.save("LUCAS")
    assert g.view()["ghost_lap_s"] == 6.0
    # faster lap but a worse score than the ghost's 7.0
    off = [(8.0, -50.0, -40.0), (8.5, 0.0, 0.0), (9.0, 100.0, 0.0), (9.5, 100.0, 100.0)]
    race_to_finish(g, path=off)
    g.save("AMY")
    assert g.view()["score"]["time_s"] == 7.5
    assert g.view()["ghost_lap_s"] == 5.5
    race_to_finish(g, path=[(5.0, 0.0, 0.0), (6.0, 100.0, 0.0), (7.0, 100.0, 100.0)])
    g.save("LUCAS")
    assert g.view()["ghost_lap_s"] == 3.0
    assert len(list((tmp_path / "data" / "runs").iterdir())) == 4


def test_mode_switches_with_the_race_and_attract():
    g = Game(CFG, MAZE, fake_step)
    assert g.mode() is None
    g.tick(rec(0.0), False, False)
    assert g.mode() is None
    g.attract = True
    assert g.mode() == "path"
    g.tick(rec(0.1, armed=False), False, False)
    assert g.mode() is None
    g.tick(rec(0.2), False, False)
    g.start()
    assert g.mode() == "hand"
    g.tick(rec(3.5), False, False)
    assert g.mode() == "hand"
    g.abort()
    assert g.mode() == "path"


def test_attract_comes_from_settings_and_is_off_by_default():
    g = Game(load(), MAZE, fake_step)
    assert g.attract is False
    g.tick(rec(0.0), False, False)
    assert g.race.state == "idle"
    assert g.mode() is None
    on = Game({**CFG, "game": {**CFG["game"], "attract": True}}, MAZE, fake_step)
    on.tick(rec(0.0), False, False)
    assert on.mode() == "path"


def test_mode_is_none_after_a_finish_or_dnf():
    g = Game(CFG, MAZE, fake_step)
    race_to_finish(g)
    assert g.race.state == "finished"
    assert g.mode() is None
    g = Game(CFG, MAZE, fake_step)
    g.tick(rec(0.0), False, False)
    g.start()
    g.tick(rec(3.0), False, False)
    g.tick(rec(3.5), True, False)
    assert g.race.state == "dnf"
    assert g.mode() is None


def test_ghost_from_records_and_from_the_settings_path(tmp_path, monkeypatch):
    g = Game(CFG, MAZE, fake_step, records(APPROACH + LAP + TAIL))
    assert g.trace[0] == [0.0, -60.0, 0.0]
    assert g.splits == [2.0, 5.0, 7.0]
    g.tick(rec(0.0), False, False)
    assert g.view()["ghost_lap_s"] == 7.0
    monkeypatch.setattr(session, "ROOT", tmp_path)
    cfg = {**CFG, "game": {**CFG["game"], "ghost": "lap.jsonl"}}
    assert Game(cfg, MAZE, fake_step).trace == []
    ghost.save(records(LAP), tmp_path / "lap.jsonl")
    g = Game(cfg, MAZE, fake_step)
    assert g.splits == [2.0, 5.0, 7.0]
    g.seed_ghost(tmp_path / "nope.jsonl")
    assert g.splits == [2.0, 5.0, 7.0]


def test_view_is_empty_before_the_first_tick():
    g = Game(CFG, MAZE, fake_step)
    assert g.view() == {}
    g.tick(rec(0.0), False, False)
    assert g.view()["state"] == "idle"
