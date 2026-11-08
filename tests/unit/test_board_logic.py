import json
import shutil
import subprocess

import pytest

from src.web.server import STATIC

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

RUN = """
const fs = require("fs"), vm = require("vm");
const box = {};
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), box);
const calls = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(calls.map(([fn, args]) => box[fn](...args))));
"""


def run(*calls):
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC / "board.js")],
        input=json.dumps(calls),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def quads(*tilts):
    return run(*[("boardQuad", [tx, ty, 8]) for tx, ty in tilts])


def test_flat_board_is_the_unit_square():
    assert quads((0, 0)) == [[[-1, -1], [1, -1], [1, 1], [-1, 1]]]


def test_tilt_x_pulls_the_plus_x_edge_in_and_shrinks_it():
    flat, small, big, neg = quads((0, 0), (0.5, 0), (1.5, 0), (-1.5, 0))
    assert flat[1][0] > small[1][0] > big[1][0]
    assert big[1][0] == big[2][0] and big[1][1] == -big[2][1]
    assert big[2][1] - big[1][1] < big[3][1] - big[0][1]
    assert -big[0][0] > 1 and big[2][1] < 1
    assert neg == [[-x, y] for x, y in (big[1], big[0], big[3], big[2])]


def test_tilt_y_pulls_the_plus_y_edge_in_and_shrinks_it():
    flat, small, big, neg = quads((0, 0), (0, 0.5), (0, 1.5), (0, -1.5))
    assert flat[2][1] > small[2][1] > big[2][1]
    assert big[2][1] == big[3][1]
    assert big[2][0] - big[3][0] < big[1][0] - big[0][0]
    assert neg == [[x, -y] for x, y in (big[3], big[2], big[1], big[0])]


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


def test_ghost_hand_shows_only_when_a_hand_is_wired_but_not_live():
    cases = {
        (False, False, False): False,
        (False, True, True): False,
        (True, False, False): True,
        (True, False, True): True,
        (True, True, True): True,
        (True, True, False): False,
    }
    calls = [
        ("ghostOn", [{"hand_seen": seen, "hand_calibrating": cal}, {"hand": hand}])
        for hand, seen, cal in cases
    ]
    assert run(*calls) == list(cases.values())
