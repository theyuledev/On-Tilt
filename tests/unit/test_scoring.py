from dataclasses import replace

import pytest

from src.games import maze, race, scoring
from src.games.progress import Progress
from tests.unit.test_contracts import make_record

pytestmark = pytest.mark.unit

GAME = {"countdown_s": 3.0, "lost_s": 1.0, "checkpoint_r_mm": 30, "max_race_s": 60}
RIG = {"plate_inner_mm": 305}
LINE = maze.Maze(
    points=((-100.0, 0.0), (0.0, 0.0), (100.0, 0.0)),
    checkpoints=(1, 2),
    start=(-100.0, 0.0),
)


def rec(t, x=-100.0, y=0.0, valid=True):
    base = make_record()
    return replace(base, t_s=t, ball=replace(base.ball, x_mm=x, y_mm=y, valid=valid, t_s=t))


def feed(path, p=None, m=LINE):
    p = p or Progress(prev_t=0.0)
    for t, x, y in path:
        p = scoring.step(p, rec(t, x, y), m, GAME, RIG)
    return p


def test_gates_count_only_in_order():
    p = feed([(1.0, 100.0, 0.0)])
    assert (p.reached, p.splits, p.prev_t) == (0, (), 1.0)
    p = feed([(2.0, 0.0, 0.0), (3.0, 100.0, 0.0)], p)
    assert (p.reached, p.splits) == (2, (2.0, 3.0))


def test_ring_edge_counts():
    assert feed([(1.0, 30.0, 0.0)]).reached == 1
    assert feed([(1.0, 30.01, 0.0)]).reached == 0


@pytest.mark.parametrize("x,y", [(152.6, 0.0), (0.0, -152.6)])
def test_out_past_the_plate_edge(x, y):
    p = feed([(1.0, x, y)])
    assert p.out and p.prev_t == 1.0 and p.reached == 0


def test_edge_itself_is_not_out():
    assert not feed([(1.0, 152.5, -152.5)]).out


def test_done_at_the_last_gate():
    p = feed([(1.0, 0.0, 0.0), (2.5, 50.0, 0.0), (4.0, 100.0, 0.0)])
    assert (p.reached, p.splits, p.done_t, p.off_s) == (2, (1.0, 4.0), 4.0, 0.0)


def test_invalid_ball_only_moves_prev_t():
    p = feed([(1.0, 0.0, 0.0)])
    q = scoring.step(p, rec(2.0, 500.0, 0.0, valid=False), LINE, GAME, RIG)
    assert q == replace(p, prev_t=2.0)


def test_nothing_changes_after_done():
    p = feed([(1.0, 0.0, 0.0), (2.0, 100.0, 0.0)])
    for r in (rec(3.0, 0.0, 0.0), rec(4.0, 500.0, 0.0)):
        assert scoring.step(p, r, LINE, GAME, RIG) == replace(p, prev_t=r.t_s)


def test_race_on_the_square_runs_to_finished():
    sq = maze.load("square")

    def adv(r, record):
        return race.advance(r, record, False, False, sq, GAME, RIG, scoring.step)

    x0, y0 = sq.start
    r = race.start(race.Race(), rec(0.0, x0, y0), False, False, sq, GAME)
    assert r.state == "countdown"
    r = adv(r, rec(3.0, x0, y0))
    assert r.state == "running"
    laps = [(4.0, x0 - 40.0, y0)]
    laps += [(5.0 + n, *sq.points[i]) for n, i in enumerate(sq.checkpoints)]
    for t, x, y in laps:
        r = adv(r, rec(t, x, y))
    assert r.state == "finished"
    assert r.progress.splits == (5.0, 6.0, 7.0, 8.0)
    assert race.elapsed(r, rec(9.0)) == 4.0
