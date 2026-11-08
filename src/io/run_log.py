import json
import time
from pathlib import Path

from src.bus import Bus
from src.io.settings import ROOT, env


class RunLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # line buffered so a ctrl-c loses nothing
        self._f = path.open("a", buffering=1)

    def write(self, rec) -> None:
        if self._f is None:
            return
        try:
            self._f.write(json.dumps(rec.to_dict()) + "\n")
        except OSError as e:
            # a dead disk must not take the control loop down with it
            print(f"run log stopped: {e}")
            self.close()

    def close(self) -> None:
        if self._f is None:
            return
        try:
            self._f.close()
        except OSError:
            pass
        self._f = None


def start_run_log(bus: Bus, root: Path = ROOT, clock=time.localtime) -> RunLog | None:
    if env("RUN_LOG") != "1":
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S", clock())
    log = RunLog(root / "data" / "runs" / f"{stamp}.jsonl")
    bus.subscribe("tick", log.write)
    return log
