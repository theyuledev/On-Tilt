import json
import shutil
import subprocess

import pytest

from src.web.server import STATIC

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

SCRIPTS = ("race", "board", "gates", "follow")
# "PEN" in the args is a canvas stand-in, a call that draws answers with what it drew
RUN = """
const fs = require("fs"), vm = require("vm");
const box = vm.createContext({ Path2D: class { constructor(d) { this.d = d; } } });
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
        ["node", "-e", RUN, *(str(STATIC / f"{f}.js") for f in SCRIPTS)],
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


def test_s_and_e_sit_beside_the_path_ends_towards_the_corner_and_stay_on_the_plate():
    assert run(
        ("letterAt", [[-112.5, 112.5], 135, 8]),
        ("letterAt", [[112.5, -112.5], 135, 8]),
        ("letterAt", [[0, 0], 135, 8]),
        ("letterAt", [[0, 0], 135, 16]),
        ("letterAt", [[130, -130], 135, 8]),
    ) == [[-127, 124], [127, -124], [14.5, 11.5], [29, 23], [127, -124]]


def test_a_course_gets_an_s_and_an_e_in_marker_strokes_and_a_lap_only_the_s():
    course = {"start": [-112.5, 112.5], "points": [[-112.5, 112.5], [0, 0], [112.5, -112.5]]}
    lap = {"start": [85, 85], "points": [[85, 85], [-85, 85], [85, 85]]}
    both, one = run(("ends", ["PEN", course, 135, 8]), ("ends", ["PEN", lap, 135, 8]))
    assert [c[1:] for c in both if c[0] == "translate"] == [[-127, 124], [127, -124]]
    s, e = (c[1]["d"] for c in both if c[0] == "stroke")
    assert s.startswith("M3 -5C") and e == "M3.5 -6H-3.5V6H3.5M-3.5 0H2.5"
    assert [c[1]["d"] for c in one if c[0] == "stroke"] == [s]


def test_the_map_boxes_what_the_follow_cam_shows_and_keeps_the_box_on_the_plate():
    mid, corner = run(
        ("followBox", [{"x": 0, "y": 0, "zoom": 2.3}, 135]),
        ("followBox", [{"x": 67, "y": 101, "zoom": 2.3}, 135]),
    )
    assert mid == pytest.approx([-50.1, -58.1, 100.1, 100.1], abs=0.05)
    assert corner == pytest.approx([17, 43, 100, 92], abs=0.2)
