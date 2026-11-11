import math
import shutil

import pytest

from tests.unit import scene3d
from tests.unit.scene3d import BALL_R, OWN_PATH, TAKE_R

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

COURSE = {"points": [[-80, 80], [80, 80], [80, -80]], "checkpoints": [1, 2], "start": [-80, 80]}
FLAT = {"x": 0, "y": 0, "zoom": 1, "tall": False}
# where a plate point, and the point r mm east of it, land on the 3D canvas
AT = "(c, x, y, r = 0) => projector(c, [0, 0], 135, 900)(x + r, y, 1).slice(0, 2)"
IDLE, RUNNING = {"state": "idle", "reached": 0}, {"state": "running", "reached": 0}


def flat(maze, s):
    return scene3d.run(("draw", ["PEN", scene3d.CFG, maze, s, FLAT]))[0]


def lines(drawn):
    return [call[1:3] for call in drawn if call[0] in ("moveTo", "lineTo")]


def arcs(drawn):
    """[x, y, radius] of every circle drawn."""
    return [call[1:4] for call in drawn if call[0] == "arc"]


def test_the_path_shown_is_the_course_except_in_path_mode_outside_a_race():
    cfg, snap = scene3d.CFG, scene3d.snap
    assert scene3d.run(
        ("shownPath", [COURSE, cfg, snap()]),
        ("shownPath", [COURSE, cfg, snap(mode="path", race=IDLE)]),
        ("shownPath", [COURSE, cfg, snap(mode="path", race=RUNNING)]),
        ("shownPath", [None, cfg, snap()]),
        ("shownPath", [None, cfg, snap(mode="path")]),
    ) == [COURSE["points"], OWN_PATH, COURSE["points"], None, OWN_PATH]


def test_path_mode_draws_its_own_path_and_leaves_the_courses_gates_and_letters_out():
    s = scene3d.snap(valid=False, mode="path", race=IDLE)
    drawn = flat(COURSE, s)
    assert all(p in lines(drawn) for p in OWN_PATH) and [80, -80] not in lines(drawn)
    assert arcs(drawn) == [] and not any(call[0] == "stroke" and len(call) > 1 for call in drawn)
    # the same in 3D: the path's corners are on the plate, no gate number hangs over it
    cam = scene3d.shots()["wide"]
    drawn = scene3d.draw(COURSE, s, cam)
    corners = scene3d.run(*[(AT, [cam, *p]) for p in OWN_PATH])
    assert all(any(math.dist(c, p) < 0.01 for p in lines(drawn)) for c in corners)
    assert not any(call[0] == "fillText" for call in drawn)


def test_gate_rings_and_the_start_ring_have_the_real_radius_in_the_flat_and_the_3d_view():
    waiting, racing = scene3d.snap(valid=False, race=IDLE), scene3d.snap(valid=False, race=RUNNING)
    drawn = flat(COURSE, waiting)
    for gate in ([80, 80], [80, -80]):
        assert [*gate, TAKE_R] in arcs(drawn)
    start = drawn.index(["arc", -80, 80, TAKE_R, 0, 2 * math.pi])
    assert ["setLineDash", [3, 3]] in drawn[start - 3 : start]
    # once the clock runs the start ring is gone, the gates keep theirs
    assert [-80, 80, TAKE_R] not in arcs(flat(COURSE, racing))
    assert [80, 80, TAKE_R] in arcs(flat(COURSE, racing))
    cam = scene3d.shots()["wide"]
    at = scene3d.run(*[(AT, [cam, *p, TAKE_R]) for p in COURSE["points"]])
    small = scene3d.run(*[(AT, [cam, *p, BALL_R * 11 / 8]) for p in COURSE["points"]])
    for s, rings in ((waiting, at), (racing, at[1:])):
        seen = lines(scene3d.draw(COURSE, s, cam))
        assert all(any(math.dist(p, q) < 0.01 for q in seen) for p in rings)
        assert not any(math.dist(p, q) < 0.01 for p in small for q in seen)
    assert not any(math.dist(at[0], q) < 0.01 for q in lines(scene3d.draw(COURSE, racing, cam)))
