from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from src.contracts import RunLogRecord
from src.games import ghost, race
from src.io.settings import ROOT


class Game:
    def __init__(
        self,
        cfg: dict,
        maze,
        step: race.Step,
        ghost_records: Sequence[RunLogRecord] = (),
        cal: Callable[[], dict | None] | None = None,
    ):
        self.cfg = cfg
        self.maze = maze
        self.step = step
        self.cal = cal
        self.off = False
        self.race = race.Race()
        self.records: list[RunLogRecord] = []
        self.last: RunLogRecord | None = None
        self.ring: RunLogRecord | None = None
        self.attract = cfg["game"]["attract"]
        self.killing = self.board_lost = False
        self.trace: list[list[float]] = []
        self.splits: list[float] = []
        try:
            if ghost_records:
                self.set_ghost(ghost_records)
            elif cfg["game"].get("ghost"):
                self.seed_ghost(ROOT / cfg["game"]["ghost"])
        except Exception as e:
            # a bad ghost file still lets you race, just alone
            print(f"ghost off: {e!r}")
            self.trace, self.splits = [], []

    def set_ghost(self, records: Sequence[RunLogRecord]) -> None:
        game, rig = self.cfg["game"], self.cfg["rig"]
        lap = ghost.lap(records, self.maze, game, rig, self.step)
        self.trace = ghost.trace(lap)
        self.splits = ghost.splits(lap, self.maze, game, rig, self.step)

    def seed_ghost(self, path: Path) -> None:
        if not path.exists():
            print(f"ghost file missing: {path}")
            return
        self.set_ghost(ghost.load(path))

    def start(self) -> None:
        if self.last is None:
            return
        game = self.cfg["game"]
        was = self.race
        cal = self.cal is not None
        self.race = race.start(was, self.last, self.killing, self.board_lost, self.maze, game, cal)
        if self.race.state != was.state:
            self.records, self.ring = [], None

    def abort(self) -> None:
        self.race = race.abort(self.race)

    def tick(self, rec: RunLogRecord, killing: bool, board_lost: bool) -> None:
        if self.off:
            return
        try:
            self._tick(rec, killing, board_lost)
        except Exception as e:
            # a race bug must not take the plate down with it
            print(f"race off: {e!r}")
            self.off = True

    def _tick(self, rec: RunLogRecord, killing: bool, board_lost: bool) -> None:
        self.last, self.killing, self.board_lost = rec, killing, board_lost
        was = self.race
        status = self.cal() if self.cal is not None else None
        game, rig = self.cfg["game"], self.cfg["rig"]
        self.race = race.advance(
            was, rec, killing, board_lost, self.maze, game, rig, self.step, status
        )
        now = self.race
        if was.state == "running" and now.t0_s is not None:
            if was.t0_s is None:
                # keep the last in-ring record so a saved race cuts like a run log
                self.records = [self.ring, rec] if self.ring is not None else [rec]
            else:
                self.records.append(rec)
        elif now.state in ("countdown", "running") and rec.ball.valid:
            self.ring = rec

    def mode(self) -> str | None:
        if self.off:
            return None
        if self.race.state == "calibrating":
            return "selfcal"
        if self.race.state in ("staging", "countdown", "running"):
            # countdown should hold the ball in the ring once a hold mode exists
            return "hand"
        if self.attract and self.last is not None and self.last.armed:
            return "path"
        return None

    def view(self) -> dict:
        if self.last is None or self.off:
            return {}
        try:
            return race.view(self.race, self.last, self.cfg["game"], self.trace, self.splits)
        except Exception as e:
            print(f"race off: {e!r}")
            self.off = True
            return {}

    def save(self, name: str) -> None:
        if self.race.state != "finished":
            return
        v = self.view()
        time_s = v["score"]["time_s"]
        ghost.save(self.records, ROOT / "data" / "runs" / f"{name}-{int(time_s * 1000)}.jsonl")
        if not self.trace or v["elapsed_s"] < self.trace[-1][0]:
            self.set_ghost(self.records)
