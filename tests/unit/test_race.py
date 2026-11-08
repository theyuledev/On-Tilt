from dataclasses import dataclass, replace

import pytest

from src.games import race
from src.games.progress import Progress, in_ring
from tests.unit.test_contracts import make_record

pytestmark = pytest.mark.unit

GAME = {
    "countdown_s": 3.0,
    "lost_s": 1.0,
    "checkpoint_r_mm": 30,
    "max_race_s": 60,
    "penalty_every_s": 1.0,
    "penalty_s": 0.5,
}
RIG = {"plate_inner_mm": 305}


@dataclass(frozen=True)
class Maze:
    points: tuple
    checkpoints: tuple
    start: tuple


MAZE = Maze(
    points=((-100.0, 0.0), (0.0, 0.0), (100.0, 0.0), (100.0, 100.0)),
    checkpoints=(1, 2, 3),
    start=(-100.0, 0.0),
)


def fake_step(p, rec, maze, game, rig):
    # gates in order, off_s while below the axis, out past the plate edge
    b = rec.ball
    if not b.valid:
        return replace(p, prev_t=rec.t_s)
    half = rig["plate_inner_mm"] / 2
    if abs(b.x_mm) > half or abs(b.y_mm) > half:
        return replace(p, out=True, prev_t=rec.t_s)
    off_s = p.off_s + (rec.t_s - p.prev_t if b.y_mm < -10 else 0.0)
    reached, splits, done = p.reached, p.splits, p.done_t
    if reached < len(maze.checkpoints) and in_ring(
        b, maze.points[maze.checkpoints[reached]], game["checkpoint_r_mm"]
    ):
        reached += 1
        splits = (*splits, rec.t_s)
        if reached == len(maze.checkpoints):
            done = rec.t_s
    return replace(p, reached=reached, off_s=off_s, splits=splits, done_t=done, prev_t=rec.t_s)


def rec(t, x=-100.0, y=0.0, valid=True, armed=True):
    base = make_record()
    ball = replace(base.ball, x_mm=x, y_mm=y, valid=valid, t_s=t)
    return replace(base, t_s=t, armed=armed, ball=ball)


def adv(r, record, killing=False, board_lost=False):
    return race.advance(r, record, killing, board_lost, MAZE, GAME, RIG, fake_step)


def countdown(t=0.0):
    return race.start(race.Race(), rec(t), False, False, MAZE, GAME)


def running(exit_t=4.0):
    # go at 3.0, ball leaves the ring at exit_t
    r = adv(countdown(), rec(3.0))
    return adv(r, rec(exit_t, x=-60.0))


def drive(r, path):
    for t, x, y in path:
        r = adv(r, rec(t, x, y))
    return r


FINISH = [
    (5.0, 0.0, 0.0),
    (6.0, 50.0, -40.0),
    (7.5, 50.0, -40.0),
    (8.0, 100.0, 0.0),
    (10.0, 100.0, 100.0),
]


def test_start_needs_armed_not_killing_valid_ball_in_ring():
    assert countdown().state == "countdown"
    assert countdown().go_s == 3.0
    assert race.start(race.Race(), rec(0.0, armed=False), False, False, MAZE, GAME).state == "idle"
    assert race.start(race.Race(), rec(0.0), True, False, MAZE, GAME).state == "idle"
    assert race.start(race.Race(), rec(0.0), False, True, MAZE, GAME).state == "idle"
    assert race.start(race.Race(), rec(0.0, valid=False), False, False, MAZE, GAME).state == "idle"
    assert race.start(race.Race(), rec(0.0, x=-60.0), False, False, MAZE, GAME).state == "idle"


def test_start_only_from_idle_finished_dnf():
    r = countdown()
    assert race.start(r, rec(1.0), False, False, MAZE, GAME) is r
    done = replace(r, state="dnf")
    assert race.start(done, rec(1.0), False, False, MAZE, GAME).state == "countdown"


def test_countdown_goes_running_exactly_at_go():
    r = adv(countdown(), rec(2.99))
    assert r.state == "countdown"
    r = adv(r, rec(3.0))
    assert r.state == "running"
    assert r.t0_s is None


def test_clock_starts_when_the_ball_leaves_the_ring():
    r = adv(countdown(), rec(3.0))
    r = adv(r, rec(3.5, x=-80.0))
    assert r.t0_s is None
    assert race.elapsed(r, rec(3.5)) == 0.0
    r = adv(r, rec(4.0, x=-60.0))
    assert r.t0_s == 4.0
    assert race.elapsed(r, rec(5.5)) == 1.5


def test_gates_count_only_in_order():
    r = drive(running(), [(5.0, 100.0, 0.0)])
    assert r.progress.reached == 0
    r = drive(r, [(6.0, 0.0, 0.0), (7.0, 100.0, 0.0)])
    assert r.progress.reached == 2
    assert r.progress.splits == (6.0, 7.0)


def test_finish_scores_time_plus_penalty():
    r = drive(running(), FINISH)
    assert r.state == "finished"
    v = race.view(r, rec(12.0), GAME, [], [])
    assert v["elapsed_s"] == 6.0
    assert v["penalty_s"] == 1.0
    assert v["score"] == {"time_s": 7.0, "penalty_s": 1.0, "checkpoints": 3}


@pytest.mark.parametrize(
    "kw,why",
    [({"killing": True}, "killed"), ({"board_lost": True}, "board lost")],
)
def test_dnf_on_kill_and_board_lost(kw, why):
    r = adv(running(), rec(5.0), **kw)
    assert (r.state, r.reason) == ("dnf", why)


def test_dnf_on_disarm_off_plate_and_time():
    assert adv(running(), rec(5.0, armed=False)).reason == "disarmed"
    assert adv(running(), rec(5.0, x=200.0)).reason == "off the plate"
    r = adv(countdown(), rec(3.0))
    assert adv(r, rec(63.0)).state == "running"
    r = adv(r, rec(63.01))
    assert (r.state, r.reason) == ("dnf", "time")


def test_ball_lost_longer_than_lost_s_ends_the_run():
    r = adv(running(), rec(5.0, x=-60.0))
    r = adv(r, rec(5.5, valid=False))
    assert r.state == "running"
    r = adv(r, rec(6.0, valid=False))
    assert r.state == "running"
    r = adv(r, rec(6.01, valid=False))
    assert (r.state, r.reason) == ("dnf", "ball lost")


def test_one_invalid_frame_changes_nothing():
    r = adv(countdown(), rec(0.5, valid=False))
    assert r.state == "countdown"
    r = adv(r, rec(1.0))
    r = adv(r, rec(3.0))
    assert adv(r, rec(3.6, valid=False)).state == "running"
    r = adv(countdown(), rec(0.5, valid=False))
    r = adv(r, rec(1.01, valid=False))
    assert (r.state, r.reason) == ("idle", "ball lost")


def test_abort_only_from_countdown_and_running():
    assert race.abort(countdown()).state == "idle"
    assert race.abort(running()).state == "idle"
    done = replace(running(), state="finished")
    assert race.abort(done) is done


def test_view_shape_and_negative_countdown_elapsed():
    v = race.view(countdown(), rec(1.0), GAME, [], [])
    keys = "state elapsed_s reached penalty_s off_path delta_s ghost_mm score reason ghost_lap_s"
    assert set(v) == set(keys.split())
    assert v["elapsed_s"] == -2.0
    assert v["score"] is None
    assert race.view(race.Race(), rec(1.0), GAME, [], [])["elapsed_s"] is None
    assert race.view(race.Race(), rec(1.0), {}, [], [])["penalty_s"] == 0.0


def test_ghost_follows_the_clock_and_loops_when_idle():
    tr = [[0.0, -60.0, 0.0], [2.0, 40.0, 0.0]]
    r = drive(running(), [(5.0, -40.0, 0.0)])
    assert race.view(r, rec(5.0), GAME, tr, [])["ghost_mm"] == (-10.0, 0.0)
    v = race.view(race.Race(), rec(3.0), GAME, tr, [])
    assert v["ghost_mm"] == (-10.0, 0.0)
    assert v["ghost_lap_s"] == 2.0
    assert race.view(race.Race(), rec(3.0), GAME, [], [])["ghost_mm"] is None


def test_delta_is_my_split_minus_the_ghost_split():
    r = drive(running(), [(5.5, 0.0, 0.0)])
    assert race.view(r, rec(5.5), GAME, [], [1.0, 3.0])["delta_s"] == 0.5
    assert race.view(r, rec(5.5), GAME, [], [2.0, 3.0])["delta_s"] == -0.5
    assert race.view(r, rec(5.5), GAME, [], [])["delta_s"] is None
    assert race.view(running(), rec(5.5), GAME, [], [1.0])["delta_s"] is None


def test_progress_defaults():
    assert Progress() == Progress(0, 0.0, None, False, (), None)
