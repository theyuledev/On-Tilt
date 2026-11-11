"""Plays ball frames through the page's camera director (static/cam.js) in node."""

import json
import subprocess
from itertools import groupby

from src.web.server import STATIC

# with a plate half width the 3D projection says where each frame's ball lands on a 900 px canvas
RUN = """
const fs = require("fs"), vm = require("vm");
const box = vm.createContext({ job: JSON.parse(fs.readFileSync(0, "utf8")) });
for (const f of process.argv.slice(1)) vm.runInContext(fs.readFileSync(f, "utf8"), box);
const play = `let st = camStart(); job.frames.map(f => {
  st = camStep(st, f);
  const P = job.half ? projector(st.cam, [0, 0], job.half, 900) : null;
  return { ...st, at: P ? P(f.ball.x, f.ball.y).slice(0, 2) : null };
})`;
console.log(JSON.stringify(vm.runInContext(play, box)));
"""
DT = 1 / 60


def frame(t, x, y, valid=True, phase="run", run_s=None, gate=None, path=None, gate_r=0) -> dict:
    """One camera input. run_s defaults to t, so a race that starts at 0 needs nothing more."""
    return {
        "t": t,
        "ball": {"x": x, "y": y, "valid": valid},
        "path": path,
        "gate": gate,
        "gateR": gate_r,
        "phase": phase,
        "runS": t if run_s is None else run_s,
    }


def play(frames, half=None) -> list[dict]:
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC / "cam.js"), str(STATIC / "lens.js")],
        input=json.dumps({"frames": list(frames), "half": half}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def shots(states) -> list[tuple[str, float, float]]:
    """Each shot in order with the time it came up and the time of its last frame."""
    runs = [list(g) for _, g in groupby(states, key=lambda s: (s["shot"], s["shotAt"]))]
    return [(r[0]["shot"], r[0]["t"], r[-1]["t"]) for r in runs]


def steps(states, key) -> list[float]:
    """How far a camera number moves from each frame to the next."""
    return [abs(b["cam"][key] - a["cam"][key]) for a, b in zip(states, states[1:], strict=False)]
