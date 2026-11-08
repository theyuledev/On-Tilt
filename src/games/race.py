from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import floor

from src.contracts import RunLogRecord
from src.games import ghost
from src.games.progress import Progress, in_ring

Step = Callable[[Progress, RunLogRecord, object, dict, dict], Progress]


@dataclass(frozen=True, slots=True)
class Race:
    state: str = "idle"
    go_s: float | None = None
    t0_s: float | None = None
    progress: Progress | None = None
    last_valid_t: float | None = None
    reason: str | None = None
    end_s: float | None = None
    cal: dict | None = None
    trimmed: bool = False


def start(
    race: Race, rec: RunLogRecord, killing: bool, board_lost: bool, maze, game: dict, cal=False
) -> Race:
    if race.state not in ("idle", "finished", "dnf"):
        return race
    if not rec.armed or killing or board_lost or not rec.ball.valid:
        return race
    if cal:
        return Race("calibrating", last_valid_t=rec.t_s)
    return _staging(Race("staging", last_valid_t=rec.t_s), rec, False, False, maze, game)


def abort(race: Race) -> Race:
    live = ("calibrating", "staging", "countdown", "running")
    return Race() if race.state in live else race


def advance(
    race: Race,
    rec: RunLogRecord,
    killing: bool,
    board_lost: bool,
    maze,
    game: dict,
    rig: dict,
    step: Step,
    status: dict | None = None,
) -> Race:
    if race.state == "calibrating":
        return _calibrating(race, rec, killing, board_lost, maze, game, status)
    if race.state == "staging":
        return _staging(race, rec, killing, board_lost, maze, game)
    if race.state == "countdown":
        return _countdown(race, rec, killing, board_lost, maze, game)
    if race.state == "running":
        return _running(race, rec, killing, board_lost, maze, game, rig, step)
    return race


def _seen(race: Race, rec: RunLogRecord, game: dict) -> tuple[Race, bool]:
    # one dropped frame is not a lost ball
    if rec.ball.valid:
        return replace(race, last_valid_t=rec.t_s), False
    return race, rec.t_s - race.last_valid_t > game["lost_s"]


def _calibrating(race, rec, killing, board_lost, maze, game, status) -> Race:
    if killing or board_lost or not rec.armed:
        return Race()
    state = status.get("state") if status else None
    if state == "failed":
        return Race(reason=status.get("msg") or "calibration failed")
    if state == "done":
        return _staging(Race("staging", last_valid_t=rec.t_s), rec, False, False, maze, game)
    if state == "waiting" and race.trimmed:
        # the pass started over, so a kill or a stale intent reset it
        return Race(reason="calibration interrupted")
    trimmed = race.trimmed or state == "trimming"
    return replace(race, cal=dict(status) if status else None, trimmed=trimmed)


def _staging(race, rec, killing, board_lost, maze, game) -> Race:
    if killing or board_lost or not rec.armed:
        return Race()
    if rec.ball.valid and in_ring(rec.ball, maze.start, game["checkpoint_r_mm"]):
        return Race("countdown", go_s=rec.t_s + game["countdown_s"], last_valid_t=rec.t_s)
    return race


def _countdown(race, rec, killing, board_lost, maze, game) -> Race:
    if killing or board_lost or not rec.armed:
        return Race()
    race, lost = _seen(race, rec, game)
    if lost:
        return Race(reason="ball lost")
    if rec.ball.valid and not in_ring(rec.ball, maze.start, game["checkpoint_r_mm"]):
        return Race(reason="left the start")
    if rec.t_s >= race.go_s:
        return replace(race, state="running")
    return race


def _running(race, rec, killing, board_lost, maze, game, rig, step) -> Race:
    def dnf(why):
        return replace(race, state="dnf", reason=why, end_s=rec.t_s)

    if killing:
        return dnf("killed")
    if board_lost:
        return dnf("board lost")
    if not rec.armed:
        return dnf("disarmed")
    race, lost = _seen(race, rec, game)
    if lost:
        return dnf("ball lost")
    if rec.t_s - race.go_s > game["max_race_s"]:
        return dnf("time")
    if race.t0_s is None:
        # the clock starts when the ball leaves the start ring
        if not rec.ball.valid or in_ring(rec.ball, maze.start, game["checkpoint_r_mm"]):
            return race
        race = replace(race, t0_s=rec.t_s, progress=Progress(prev_t=rec.t_s))
    progress = step(race.progress, rec, maze, game, rig)
    race = replace(race, progress=progress)
    if progress.out:
        return dnf("off the plate")
    if progress.done_t is not None:
        return replace(race, state="finished")
    return race


def elapsed(race: Race, rec: RunLogRecord) -> float | None:
    if race.state in ("idle", "calibrating", "staging"):
        return None
    if race.state == "countdown":
        return rec.t_s - race.go_s
    if race.t0_s is None:
        return 0.0
    if race.state == "finished":
        return race.progress.done_t - race.t0_s
    if race.state == "dnf":
        return race.end_s - race.t0_s
    return rec.t_s - race.t0_s


def _penalty(progress: Progress | None, game: dict) -> float:
    every, each = game.get("penalty_every_s"), game.get("penalty_s")
    if progress is None or every is None or each is None:
        return 0.0
    return floor(progress.off_s / every) * each


def _delta(race: Race, splits: list[float]) -> float | None:
    p = race.progress
    if p is None or not p.splits or len(p.splits) > len(splits):
        return None
    return (p.splits[-1] - race.t0_s) - splits[len(p.splits) - 1]


def _ghost(race: Race, rec: RunLogRecord, trace: list[list[float]], t: float | None):
    if not trace:
        return None
    lap = trace[-1][0]
    # a ghost you beat stays where it was at your finish, otherwise it loops
    if race.state == "running" or (race.state == "finished" and t < lap):
        return ghost.at(trace, t)
    if race.state in ("idle", "dnf", "finished") and lap > 0:
        return ghost.at(trace, rec.t_s % lap)
    return None


def view(
    race: Race, rec: RunLogRecord, game: dict, trace: list[list[float]], splits: list[float]
) -> dict:
    t = elapsed(race, rec)
    penalty = _penalty(race.progress, game)
    reached = race.progress.reached if race.progress else 0
    score = None
    if race.state == "finished":
        score = {"time_s": t + penalty, "penalty_s": penalty, "checkpoints": reached}
    return {
        "state": race.state,
        "elapsed_s": t,
        "reached": reached,
        "penalty_s": penalty,
        "off_path": None,
        "delta_s": _delta(race, splits),
        "ghost_mm": _ghost(race, rec, trace, t),
        "score": score,
        "reason": race.reason,
        "ghost_lap_s": trace[-1][0] if trace else None,
        "cal": race.cal,
    }
