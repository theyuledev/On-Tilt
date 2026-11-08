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


def start(race: Race, rec: RunLogRecord, killing: bool, board_lost: bool, maze, game: dict) -> Race:
    if race.state not in ("idle", "finished", "dnf"):
        return race
    if not rec.armed or killing or board_lost or not rec.ball.valid:
        return race
    if not in_ring(rec.ball, maze.start, game["checkpoint_r_mm"]):
        return race
    return Race("countdown", go_s=rec.t_s + game["countdown_s"], last_valid_t=rec.t_s)


def abort(race: Race) -> Race:
    return Race() if race.state in ("countdown", "running") else race


def advance(
    race: Race,
    rec: RunLogRecord,
    killing: bool,
    board_lost: bool,
    maze,
    game: dict,
    rig: dict,
    step: Step,
) -> Race:
    if race.state == "countdown":
        return _countdown(race, rec, killing, board_lost, game)
    if race.state == "running":
        return _running(race, rec, killing, board_lost, maze, game, rig, step)
    return race


def _seen(race: Race, rec: RunLogRecord, game: dict) -> tuple[Race, bool]:
    # one dropped frame is not a lost ball
    if rec.ball.valid:
        return replace(race, last_valid_t=rec.t_s), False
    return race, rec.t_s - race.last_valid_t > game["lost_s"]


def _countdown(race, rec, killing, board_lost, game) -> Race:
    if killing or board_lost or not rec.armed:
        return Race()
    race, lost = _seen(race, rec, game)
    if lost:
        return Race(reason="ball lost")
    if rec.t_s >= race.go_s:
        return replace(race, state="running")
    return race


def _running(race, rec, killing, board_lost, maze, game, rig, step) -> Race:
    def dnf(why):
        return replace(race, state="dnf", reason=why)

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
    if race.state == "idle":
        return None
    if race.state == "countdown":
        return rec.t_s - race.go_s
    if race.t0_s is None:
        return 0.0
    if race.state == "finished":
        return race.progress.done_t - race.t0_s
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
    if race.state in ("running", "finished"):
        return ghost.at(trace, t)
    lap = trace[-1][0]
    if race.state in ("idle", "dnf") and lap > 0:
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
    }
