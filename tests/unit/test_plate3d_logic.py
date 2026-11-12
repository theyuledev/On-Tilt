import json
import math
import shutil
import subprocess

import pytest

from src.web.server import STATIC
from tests.unit import scene3d

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
        ["node", "-e", RUN, str(STATIC / "lens.js")],
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


# one stroke between the camera and a ball in the middle of the plate
WALLED = {
    "points": [[-100, 0], [100, 0]],
    "checkpoints": [1],
    "start": [-100, 0],
    "walls": [[[-60, -30], [60, -30]]],
}


def test_the_ball_is_painted_after_everything_that_stands():
    for name, cam in scene3d.shots().items():
        drawn = scene3d.draw(WALLED, scene3d.snap(), cam)
        ball = drawn.index(["fillStyle=", "gradient"])
        # walls, rim and slab are the closed shapes, the tag after the ball is a round box
        assert max(i for i, call in enumerate(drawn) if call[0] == "closePath") < ball, name


def test_the_position_tag_never_meets_the_ball_and_stays_on_the_canvas():
    w, hgt = 220, 28
    balls = [[x, y, rad] for x in range(0, 901, 100) for y in range(0, 901, 100) for rad in (8, 45)]
    for (_, sy, rad), (x, y) in zip(
        balls, scene3d.run(*[("tagAt", [b, w, hgt, PX]) for b in balls]), strict=True
    ):
        assert w / 2 <= x <= PX - w / 2 and hgt / 2 <= y <= PX - hgt / 2
        assert abs(y - sy) - hgt / 2 > rad
    # straight down the tag used to land on the ball
    at = (40, -30)
    drawn = scene3d.draw(None, scene3d.snap(ball=at), scene3d.shots(at)["tower"])
    sx, sy, rad = next(call[1:4] for call in reversed(drawn) if call[0] == "arc")
    left, top, wide, tall = next(call[1:5] for call in drawn if call[0] == "roundRect")
    nearest = (min(max(sx, left), left + wide), min(max(sy, top), top + tall))
    assert math.dist(nearest, (sx, sy)) > rad


def test_the_rim_stands_up_round_the_board():
    wide = scene3d.shots()["wide"]
    drawn = scene3d.draw(None, scene3d.snap(valid=False), wide)
    (rim,) = scene3d.run(("() => RIM_PX", []))
    points = [call[1:3] for call in drawn if call[0] in ("moveTo", "lineTo")]
    tops = project(wide, [(x, y, rim) for x in (-HALF, HALF) for y in (-HALF, HALF)])
    assert rim > 0
    for x, y, *_ in tops:
        assert any(math.dist((x, y), p) < 0.01 for p in points)


def test_walls_stand_well_below_the_top_of_the_ball_in_every_shot():
    (wall,) = scene3d.run(("() => WALL_PX", []))
    r, up = scene3d.BALL_R, scene3d.BALL_R * 270 / HALF
    for name, cam in scene3d.shots().items():
        (_, top, _, _), (_, mid, _, k) = project(cam, [(0, -r, wall), (0, 0, up)])
        # how far up the ball a wall touching its near side reaches on screen
        assert (mid + up * k - top) / (2 * up * k) < 0.25, name


def test_a_wall_is_drawn_that_high_with_a_grey_top_not_the_red_of_the_crosshair():
    wall, side, grey, red = scene3d.run(("() => [WALL_PX, WALL_SIDE, WALL_TOP, CROSS]", []))[0]
    wide = scene3d.shots()["wide"]
    drawn = scene3d.draw(WALLED, scene3d.snap(), wide)
    after = drawn[drawn.index(["fillStyle=", side]) :]
    quad = [call[1:3] for call in after if call[0] in ("moveTo", "lineTo")][:4]
    for got, want in zip(quad[2:], project(wide, [(60, -30, wall), (-60, -30, wall)]), strict=True):
        assert got == pytest.approx(want[:2])
    assert next(call[1] for call in after if call[0] == "strokeStyle=") == grey != red


def test_the_tag_text_stays_big_enough_to_read_on_a_small_canvas():
    (least,) = scene3d.run(("() => TAG_MIN_PX", []))
    tower = scene3d.shots()["tower"]
    for px, want in ((300, least), (PX, 23)):
        drawn = scene3d.draw(None, scene3d.snap(), tower, px=px)
        box = next(i for i, call in enumerate(drawn) if call[0] == "roundRect")
        font = next(call[1] for call in reversed(drawn[:box]) if call[0] == "font=")
        assert float(font.split("px")[0]) == pytest.approx(want)


def test_the_tray_casts_a_soft_shadow_that_stays_under_the_board():
    drawn = scene3d.draw(None, scene3d.snap(valid=False), scene3d.shots()["wide"])
    names = [call[0] for call in drawn]
    blur, fills = names.index("shadowBlur="), [i for i, n in enumerate(names) if n == "fill"]
    assert drawn[blur][1] > 0 and names.index("save") < blur < fills[0]
    # switched off again before anything else is painted
    assert fills[0] < names.index("restore") < fills[1]
