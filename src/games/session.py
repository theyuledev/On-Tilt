from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from src.contracts import RunLogRecord
from src.games import ghost, race
from src.io.settings import ROOT


class Game:
    def __init__(
        self, cfg: dict, maze, step: race.Step, ghost_records: Sequence[RunLogRecord] = ()
    ):
        self.cfg = cfg
        self.maze = maze
        self.step = step
        self.race = race.Race()
        self.records: list[RunLogRecord] = []
        self.last: RunLogRecord | None = None
        self.attract = False
        self.killing = self.board_lost = False
        self.trace: list[list[float]] = []
        self.splits: list[float] = []
        if ghost_records:
            self.set_ghost(ghost_records)
        elif cfg["game"].get("ghost"):
            self.seed_ghost(ROOT / cfg["game"]["ghost"])

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
        self.race = race.start(was, self.last, self.killing, self.board_lost, self.maze, game)
        if self.race.state == "countdown" and was.state != "countdown":
            self.records = []

    def abort(self) -> None:
        self.race = race.abort(self.race)

    def tick(self, rec: RunLogRecord, killing: bool, board_lost: bool) -> None:
        self.last, self.killing, self.board_lost = rec, killing, board_lost
        was = self.race
        self.race = race.advance(
            was, rec, killing, board_lost, self.maze, self.cfg["game"], self.cfg["rig"], self.step
        )
        if was.state == "running" and self.race.t0_s is not None:
            self.records.append(rec)

    def mode(self) -> str:
        if self.race.state in ("countdown", "running"):
            # countdown should hold the ball in the ring once a hold mode exists
            return "hand"
        if self.attract and self.last is not None and self.last.armed:
            return "path"
        return "hand"

    def view(self) -> dict:
        if self.last is None:
            return {}
        return race.view(self.race, self.last, self.cfg["game"], self.trace, self.splits)

    def save(self, name: str) -> None:
        if self.race.state != "finished":
            return
        time_s = self.view()["score"]["time_s"]
        ghost.save(self.records, ROOT / "data" / "runs" / f"{name}-{int(time_s * 1000)}.jsonl")
