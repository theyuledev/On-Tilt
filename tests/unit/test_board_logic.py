import json
import shutil
import subprocess

import pytest

from src.web.server import STATIC

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

# "PEN" in the args is a canvas stand-in, a call that draws answers with what it drew
RUN = """
const fs = require("fs"), vm = require("vm");
const box = vm.createContext({});
for (const f of process.argv.slice(1)) vm.runInContext(fs.readFileSync(f, "utf8"), box);
const calls = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(calls.map(([fn, args]) => {
  const drawn = [];
  const pen = new Proxy({}, { get: (t, k) => (...a) => drawn.push([k, ...a]), set: () => true });
  const out = box[fn](...args.map(a => (a === "PEN" ? pen : a)));
  return args.includes("PEN") ? drawn : out;
})));
"""


def run(*calls):
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC / "race.js"), str(STATIC / "board.js")],
        input=json.dumps(calls),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_guard_is_null_until_the_snapshot_has_one_and_reads_weight_and_side():
    shape = {"w": [0.3, "x"], "side": [-2, 0]}
    assert run(
        ("guardOf", [{}]),
        ("guardOf", [{"guard": None}]),
        ("guardOf", [{"guard": {"w": 0.5}}]),
        ("guardOf", [{"guard": shape}]),
    ) == [None, None, None, {"w": [0.3, 0], "side": [-1, 0]}]


def test_guard_glow_is_off_below_a_trace_then_clamped():
    ws = [0, 0.04, 0.05, 0.1, 0.4, 0.9]
    assert run(*[("glowAlpha", [w]) for w in ws]) == [0, 0, 0.15, 0.15, 0.4, 0.6]


def test_guard_line_needs_a_guard_and_runs_13_mm_inside_the_rim_with_the_held_side_solid():
    held = {"guard": {"w": [0.4, 0.0], "side": [1, -1]}}
    none, drawn = run(("guardLine", ["PEN", {}, 135]), ("guardLine", ["PEN", held, 135]))
    assert none == []
    assert ["setLineDash", [2, 3]] in drawn and ["strokeRect", -122, -122, 244, 244] in drawn
    lines = [c for c in drawn if c[0] in ("moveTo", "lineTo")]
    assert lines == [["moveTo", 122, -122], ["lineTo", 122, 122]]
