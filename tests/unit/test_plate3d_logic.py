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
const box = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), box);
const asked = JSON.parse(fs.readFileSync(0, "utf8"));
console.log(JSON.stringify(asked.map(js => vm.runInContext(js, box))));
"""
PX, HALF = 900, 135
FLAT = [0, 0]


def project(cam, points, tilt=FLAT):
    """[canvas x, canvas y, depth, scale] of each plate point (x mm, y mm, height in scene px)."""
    shot = {"pitch": 0, "yaw": 0, "zoom": 1, "cx": 0, "cy": 0, **cam}
    make = f"projector({json.dumps(shot)}, {json.dumps(tilt)}, {HALF}, {PX})"
    asked = [f"{make}({x}, {y}, {z})" for x, y, z in points]
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC / "plate3d.js")],
        input=json.dumps(asked),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_straight_down_the_plate_is_a_square_with_x_right_and_y_up_the_screen():
    mid, right, far, raised = project({}, [(0, 0, 0), (HALF, 0, 0), (0, HALF, 0), (0, 0, 40)])
    assert mid[:2] == [PX / 2, 0.54 * PX]
    # the plate's half width is 270 scene px, the canvas shows 900 of them
    assert right[0] - mid[0] == pytest.approx(270) and right[1] == pytest.approx(mid[1])
    assert mid[1] - far[1] == pytest.approx(270) and far[0] == pytest.approx(mid[0])
    # looking straight down a taller thing is just nearer, so it draws bigger
    assert raised[:2] == pytest.approx(mid[:2]) and raised[2] < mid[2] and raised[3] > mid[3]


def test_the_camera_looks_at_its_centre_and_zoom_scales_about_it():
    at, off = project({"cx": 50, "cy": -20, "zoom": 2}, [(50, -20, 0), (60, -20, 0)])
    assert at[:2] == [PX / 2, 0.54 * PX]
    assert off[0] - at[0] == pytest.approx(2 * 10 * 270 / HALF)


def test_yaw_turns_the_heading_up_the_screen():
    mid, east = project({"yaw": 90}, [(0, 0, 0), (HALF, 0, 0)])
    assert east[0] == pytest.approx(mid[0]) and east[1] < mid[1]


def test_pitch_lays_the_plate_back_far_side_smaller_and_walls_standing_up():
    cam = {"pitch": 60}
    mid, far, near, top = project(cam, [(0, 0, 0), (0, HALF, 0), (0, -HALF, 0), (0, 0, 24)])
    assert far[1] < mid[1] < near[1]
    assert far[2] > mid[2] > near[2] and far[3] < mid[3] < near[3]
    # the far half takes less of the screen than the near half
    assert mid[1] - far[1] < near[1] - mid[1]
    assert top[0] == pytest.approx(mid[0]) and top[1] < mid[1]
    left, right = project(cam, [(-HALF, HALF, 0), (HALF, HALF, 0)])
    assert right[0] - left[0] < 2 * 270


def test_a_positive_tilt_drops_that_axis_plus_edge():
    cam = {"pitch": 60}
    flat = project(cam, [(HALF, 0, 0), (-HALF, 0, 0), (0, HALF, 0)])
    leant = project(cam, [(HALF, 0, 0), (-HALF, 0, 0), (0, HALF, 0)], tilt=[2, 0])
    assert leant[0][1] > flat[0][1] and leant[1][1] < flat[1][1]
    assert leant[2][:2] == pytest.approx(flat[2][:2])
    back = project(cam, [(0, HALF, 0)], tilt=[0, 2])
    assert back[0][1] > flat[2][1]


def test_a_point_behind_the_eye_stays_a_finite_number():
    out = project({"pitch": 72, "zoom": 2.4, "cy": 135}, [(0, -135, 60)])
    assert all(isinstance(v, int | float) and abs(v) < 1e6 for v in out[0])
