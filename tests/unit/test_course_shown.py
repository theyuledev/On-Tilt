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
STAGING, FINISHED = {"state": "staging", "reached": 0}, {"state": "finished", "reached": 2}
# a lap ends on its start; a longer course to tell taken, next and later gates apart
LAP = {**COURSE, "points": [*COURSE["points"], [-80, 80]], "checkpoints": [1, 2, 3]}
LONG = {**COURSE, "points": [*COURSE["points"], [-80, -80]], "checkpoints": [1, 2, 3]}
INK = "() => [PATH, GATE, GATE_LIT]"
# canvas px per scene px where the camera looks, what the 3D line widths scale with
K = "c => projector(c, [0, 0], 135, 900)(c.cx, c.cy)[3]"


def flat(maze, s):
    return scene3d.run(("draw", ["PEN", scene3d.CFG, maze, s, FLAT]))[0]


def last(drawn, i, name):
    """What `name` was last set to, or called with, before call i."""
    return next(call[1] for call in reversed(drawn[:i]) if call[0] == name)


def ring_at(drawn, at):
    """Where the 3D rings start whose east point lands on `at`."""
    return [i for i, c in enumerate(drawn) if c[0] == "moveTo" and math.dist(c[1:3], at) < 0.01]


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
        ("shownPath", [COURSE, cfg, snap(mode="path", race=STAGING)]),
        ("shownPath", [COURSE, cfg, snap(mode="path", race=FINISHED)]),
        ("shownPath", [None, cfg, snap()]),
        ("shownPath", [None, cfg, snap(mode="path")]),
    ) == [COURSE["points"], OWN_PATH, COURSE["points"], COURSE["points"], OWN_PATH, None, OWN_PATH]


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
    assert last(drawn, start, "lineWidth=") == 0.8
    # the small numbered ring stays in the middle of the big one
    assert [80, 80, BALL_R * 11 / 8] in arcs(drawn)
    assert any(call[:2] == ["fillText", "1"] for call in drawn)
    # once the clock runs the start ring is gone for good, the gates keep theirs
    assert [-80, 80, TAKE_R] not in arcs(flat(COURSE, racing))
    assert [-80, 80, TAKE_R] not in arcs(flat(COURSE, scene3d.snap(valid=False, race=FINISHED)))
    assert [80, 80, TAKE_R] in arcs(flat(COURSE, racing))
    cam = scene3d.shots()["wide"]
    at = scene3d.run(*[(AT, [cam, *p, TAKE_R]) for p in COURSE["points"]])
    small = scene3d.run(*[(AT, [cam, *p, BALL_R * 11 / 8]) for p in COURSE["points"]])
    for s, rings in ((waiting, at), (racing, at[1:])):
        seen = lines(scene3d.draw(COURSE, s, cam))
        assert all(any(math.dist(p, q) < 0.01 for q in seen) for p in rings)
        assert not any(math.dist(p, q) < 0.01 for p in small for q in seen)
    assert not any(math.dist(at[0], q) < 0.01 for q in lines(scene3d.draw(COURSE, racing, cam)))


def test_a_laps_start_ring_lies_over_its_last_gate_in_the_paths_colour_while_the_ball_is_wanted():
    path, gate, _ = scene3d.run((INK, []))[0]
    ring, cam = ["arc", -80, 80, TAKE_R, 0, 2 * math.pi], scene3d.shots()["wide"]
    [east] = scene3d.run((AT, [cam, -80, 80, TAKE_R]))
    for race, ink in ((STAGING, path), ({"state": "countdown", "reached": 0}, path), (IDLE, gate)):
        s = scene3d.snap(valid=False, race=race)
        drawn = flat(LAP, s)
        top = max(i for i, call in enumerate(drawn) if call == ring)
        assert drawn.count(ring) == 2
        assert last(drawn, top, "setLineDash") == [3, 3] and last(drawn, top, "strokeStyle=") == ink
        assert next((c[1] for c in drawn[top:] if c[0] == "setLineDash"), None) == []
        drawn = scene3d.draw(LAP, s, cam)
        both = ring_at(drawn, east)
        assert len(both) == 2
        assert last(drawn, both[1], "setLineDash") != []
        assert last(drawn, both[1], "strokeStyle=") == ink
        # and the dashes end with the ring
        assert next((c[1] for c in drawn[both[1] :] if c[0] == "setLineDash"), None) == []


def test_take_rings_tell_taken_next_and_later_gates_apart_in_both_views():
    inks, gates = scene3d.run((INK, []))[0], LONG["points"][1:]
    inks = [inks[2], inks[0], inks[1]]
    s = scene3d.snap(valid=False, race={"state": "running", "reached": 1})
    drawn = flat(LONG, s)
    rings = [drawn.index(["arc", *p, TAKE_R, 0, 2 * math.pi]) for p in gates]
    assert [last(drawn, i, "strokeStyle=") for i in rings] == inks
    cam = scene3d.shots()["wide"]
    drawn, k = scene3d.draw(LONG, s, cam), scene3d.run((K, [cam]))[0]
    for at, ink in zip(scene3d.run(*[(AT, [cam, *p, TAKE_R]) for p in gates]), inks, strict=True):
        [i] = ring_at(drawn, at)
        assert last(drawn, i, "strokeStyle=") == ink and last(drawn, i, "setLineDash") == []
        assert last(drawn, i, "lineWidth=") == pytest.approx(2 * k)
