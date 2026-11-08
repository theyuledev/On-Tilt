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
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(cases.map(([s, ctx]) => box.guide(s, ctx))));
"""

ON = {"live": True, "watch": False, "hand": True, "lost": False}
BALL = {"ball": {"valid": True}}
ARMED = {**BALL, "armed": True, "hand_seen": True}


def guide(*cases):
    cases = [(s, {**ON, **ctx}) for s, ctx in cases]
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC / "guide.js")],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return [(g["text"], g["tone"]) for g in json.loads(out.stdout)]


def race(state, **kw):
    return {**ARMED, "race": {"state": state, **kw}}


def test_each_branch_says_one_next_step():
    cases = {
        ("connecting...", "warn"): ({**ARMED, "board_lost": True}, {"live": False}),
        ("board lost, check the usb cable", "stop"): ({"board_lost": True}, {}),
        ("camera lost", "stop"): ({"camera_lost": True}, {}),
        ("stopped", "stop"): ({"killing": True}, {}),
        ("", ""): ({}, {"watch": True}),
        ("mode needs a hand", "warn"): ({"arm_blocked": "mode needs a hand"}, {}),
        ("no input, plate levelled", "warn"): ({**ARMED, "stale": True}, {}),
        ("HAND LOST", "stop"): ({**ARMED, "hand_seen": False}, {"lost": True}),
        ("calibrating... keep your hand off, place the ball in the middle", "warn"): (
            race("calibrating"),
            {},
        ),
        ("2 of 3", "warn"): (race("calibrating", cal={"msg": "2 of 3"}), {}),
        ("roll the ball into the start ring", ""): (race("staging"), {}),
        ("put the ball on the plate", ""): ({"ball": {"valid": False}}, {}),
        ("click arm to start", ""): (BALL, {}),
        ("arrows / wasd to tilt, r to race", ""): (ARMED, {"hand": False}),
        ("hold still... locking neutral", "warn"): ({**ARMED, "hand_calibrating": True}, {}),
        ("hand flat under the webcam, palm down", ""): ({**ARMED, "hand_seen": False}, {}),
        ("tilt your hand to roll the ball, r to race", ""): (ARMED, {}),
    }
    assert guide(*cases.values()) == list(cases)


def test_hud_race_states_leave_the_banner_empty():
    states = ("countdown", "running", "finished", "dnf")
    assert guide(*[(race(st), {}) for st in states]) == [("", "")] * 4


def test_stop_states_win_in_order_and_watch_still_sees_them():
    lost = {**ARMED, "hand_seen": False}
    assert guide(
        ({**lost, "killing": True, "race": {"state": "running"}}, {"lost": True}),
        ({**lost, "board_lost": True, "camera_lost": True, "killing": True}, {"lost": True}),
        ({"camera_lost": True}, {"watch": True}),
        (lost, {"watch": True, "lost": True}),
        (None, {}),
        ({**ARMED, "hand_calibrating": True, "stale": True}, {}),
    ) == [
        ("stopped", "stop"),
        ("board lost, check the usb cable", "stop"),
        ("camera lost", "stop"),
        ("", ""),
        ("put the ball on the plate", ""),
        ("hold still... locking neutral", "warn"),
    ]
