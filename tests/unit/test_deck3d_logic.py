import math
import shutil

import pytest

from tests.unit import scene3d
from tests.unit.scene3d import BALL_R, HALF, TAKE_R, frames, joined, lines

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

COURSE = {"points": [[-80, 80], [80, 80], [80, -80]], "checkpoints": [1, 2], "start": [-80, 80]}
LAP = {"points": [[-80, 80], [80, 80], [-80, 80]], "checkpoints": [1, 2], "start": [-80, 80]}
# where a plate point lands on the 3D canvas
AT = "(c, x, y) => projector(c, [0, 0], 135, 900)(x, y, 1).slice(0, 2)"
LENS = "const P = projector(cam, [0, 0], 135, 900)"


def at(cam, *points):
    return scene3d.run(*[(AT, [cam, *p]) for p in points])


def has(drawn, point):
    return any(math.dist(point, p) < 0.01 for p in lines(drawn))


def race(state, **kw):
    return {"state": state, "reached": 0, **kw}


def test_the_flat_pen_sends_every_point_through_the_lens_and_scales_widths_and_dashes():
    cam = scene3d.shots()["chase"]
    job = f"""(pen, cam) => {{ {LENS}, p = flatPen(pen, P, 2);
      p.lineWidth = 3; p.setLineDash([3, 1]); p.strokeStyle = "#123456";
      polyline(p, [[0, 0], [50, 20]]); circle(p, [10, 10], 5); p.strokeRect(-10, -10, 20, 20); }}"""
    (drawn,) = scene3d.run((job, ["PEN", cam]))
    assert ["lineWidth=", 6] in drawn and ["setLineDash", [6, 2]] in drawn
    assert ["strokeStyle=", "#123456"] in drawn
    mm = [[0, 0], [50, 20], [15, 10], [10, 15], [-10, -10], [10, -10], [10, 10], [-10, 10]]
    assert all(has(drawn, p) for p in at(cam, *mm))
    # nothing reaches the canvas in plate mm, and the rectangle is stroked as a canvas would
    assert not any(call[0] in ("arc", "strokeRect") for call in drawn)
    assert drawn[-1] == ["stroke"]


def test_the_flat_pen_calls_the_canvas_as_the_canvas_like_a_browser_demands():
    job = f"""cam => {{ const seen = [], real = {{}}, guard = n => {{
        if (n !== real) throw new Error("illegal invocation"); }};
      for (const m of "beginPath moveTo lineTo closePath stroke fill setLineDash".split(" "))
        real[m] = function () {{ guard(this); seen.push(m); }};
      let w = 0;
      const width = {{ set(v) {{ guard(this); w = v; }}, get: () => w }};
      Object.defineProperty(real, "lineWidth", width);
      {LENS}, p = flatPen(real, P, 2);
      p.lineWidth = 3; p.setLineDash([1, 1]); polyline(p, [[0, 0], [10, 0]]); circle(p, [0, 0], 5);
      p.fill(); return [w, p.lineWidth, seen.includes("stroke") && seen.includes("fill")]; }}"""
    assert scene3d.run((job, [scene3d.shots()["wide"]])) == [[6, 6, True]]


def test_a_decal_puts_plate_mm_on_the_tilted_plate_exact_at_its_spot_and_close_round_it():
    spot = [40, -30]
    for name, cam in scene3d.shots(spot).items():
        job = f"(pen, cam) => {{ {LENS}; decal(pen, P, [40, -30], () => 0); }}"
        (drawn,) = scene3d.run((job, ["PEN", cam]))
        a, b, c, d, e, f = next(call[1:] for call in drawn if call[0] == "setTransform")
        names = [call[0] for call in drawn]
        assert names.index("save") < names.index("setTransform") < names.index("restore"), name
        for off, slack in ((0, 0.001), (8, 1.0), (11, 2.5)):
            mm = [[spot[0] + sx * off, spot[1] + sy * off] for sx in (-1, 1) for sy in (-1, 1)]
            for (x, y), true in zip(mm, at(cam, *mm), strict=True):
                assert math.dist((a * x + c * y + e, b * x + d * y + f), true) <= slack, (name, off)
        # +y on the plate is up the screen, so the flat view's upright text stays upright
        assert d < 0 < a, name


def test_the_path_splits_at_the_ball_between_the_last_gate_and_the_next():
    path, ball = COURSE["points"], lambda x, y, ok=True: {"x_mm": x, "y_mm": y, "valid": ok}
    split = "pathSplit"
    cuts = scene3d.run(
        (split, [path, COURSE, race("idle"), ball(0, 80)]),
        (split, [path, COURSE, race("finished", reached=2), ball(80, -80)]),
        (split, [path, COURSE, race("running"), ball(0, 70)]),
        (split, [path, COURSE, race("running", reached=1), ball(75, 0)]),
        # past gate 1 a ball that drifts back beside the first leg still counts on the second
        (split, [path, COURSE, race("running", reached=1), ball(60, 79)]),
        (split, [path, COURSE, race("running"), ball(0, 70, False)]),
        # before gate 1 a ball beside the second leg, or past the corner, stays on the first
        (split, [path, COURSE, race("running"), ball(79, 0)]),
        (split, [path, COURSE, race("running"), ball(120, 80)]),
        # a stretch with no length has nowhere to cut
        (split, [[[0, 0], [0, 0]], {"checkpoints": [1]}, race("running"), ball(5, 5)]),
    )
    idle, done, first, second, beside, lost, early, past, flat = cuts
    assert idle == [[], path] and lost == [[], path] and done == [path, []]
    assert first == [[[-80, 80], [0, 80]], [[0, 80], [80, 80], [80, -80]]]
    assert second == [[[-80, 80], [80, 80], [80, 0]], [[80, 0], [80, -80]]]
    assert beside == [[[-80, 80], [80, 80], [80, 79]], [[80, 79], [80, -80]]]
    assert early == [[[-80, 80], [79, 80]], [[79, 80], [80, 80], [80, -80]]]
    assert past == [[[-80, 80], [80, 80]], [[80, 80], [80, 80], [80, -80]]]
    assert flat == [[], [[0, 0], [0, 0]]]


def test_the_3d_plate_carries_s_e_and_the_numbered_gates_and_no_floating_signs():
    cam = scene3d.shots()["wide"]
    drawn = scene3d.draw(COURSE, scene3d.snap(valid=False, race=race("idle")), cam)
    letters = [call for call in drawn if call[0] == "stroke" and len(call) > 1]
    assert len(letters) == 2 and [call[1] for call in drawn if call[0] == "fillText"] == ["1"]
    # each sits in a decal at its own place, the number at the gate
    plain = [1, 0, 0, 1, 0, 0]
    spots = [call[1:] for call in drawn if call[0] == "setTransform" and call[1:] != plain]
    (gate,) = at(cam, [80, 80])
    landed = [(a * 80 + c * 80 + e, b * 80 + d * 80 + f) for a, b, c, d, e, f in spots]
    assert len(spots) == 4 and any(math.dist(p, gate) < 0.01 for p in landed)
    lap = scene3d.draw(LAP, scene3d.snap(valid=False, race=race("idle")), cam)
    assert len([call for call in lap if call[0] == "stroke" and len(call) > 1]) == 1


def test_the_3d_path_is_a_ribbon_with_the_driven_part_told_apart():
    cam, s = scene3d.shots()["wide"], scene3d.snap(ball=(0, 78), race=race("running"))
    drawn = scene3d.draw(COURSE, s, cam)
    wide, thin, blue, green = scene3d.run(("() => [RIBBON_MM, DONE_MM, PATH, GATE_LIT]", []))[0]
    # canvas px per plate mm where the camera looks
    (mm,) = scene3d.run(("c => projector(c, [0, 0], 135, 900)(c.cx, c.cy)[3] * 270 / 135", [cam]))
    (cut,), widths = at(cam, [0, 80]), [call[1] for call in drawn if call[0] == "lineWidth="]
    assert wide > thin > 0 and any(w == pytest.approx(wide * mm) for w in widths)
    assert any(w == pytest.approx(thin * mm) for w in widths)
    # the ribbon is cut where the ball is: that point is drawn, and both colours are used
    assert has(drawn, cut) and ["strokeStyle=", blue] in drawn and ["strokeStyle=", green] in drawn
    # the driven part is a line of its own, from the start to the ball
    i = drawn.index(["strokeStyle=", green])
    assert joined(drawn[i : i + drawn[i:].index(["stroke"])], at(cam, [-80, 80])[0], cut)
    # and no longer the old dashed line
    assert [call[1] for call in drawn if call[0] == "setLineDash" and call[1]] == []


def test_the_3d_view_keeps_the_trail_and_shows_the_guard_line_and_the_whole_target():
    cam = scene3d.shots()["wide"]
    first, second = scene3d.snap(ball=(0, 0)), {**scene3d.snap(ball=(20, 10)), "t_s": 1.1}
    assert joined(frames(None, cam, first, second), *at(cam, [0, 0], [20, 10]))
    guarded = {**scene3d.snap(valid=False), "guard": {"w": [0.4, 0], "side": [1, -1]}}
    edge = HALF - 13
    drawn = scene3d.draw(None, guarded, cam)
    assert all(has(drawn, p) for p in at(cam, [edge, edge], [-edge, -edge]))
    aimed = scene3d.snap(valid=False, mode="balance")
    aimed["target"] = {"src": "mouse", "mm": [30, -20], "found": True}
    drawn = scene3d.draw(None, aimed, cam)
    marks = [[30 + n * BALL_R, -20] for n in (1.25, 3.125, 4)] + [[30, -20 - 4 * BALL_R]]
    assert all(has(drawn, p) for p in at(cam, *marks))


def test_both_views_keep_the_ghosts_trail_and_drop_it_when_the_ghost_goes():
    cam = scene3d.shots()["wide"]

    def frame(t, ball, ghost):
        s = scene3d.snap(ball=ball, race=race("running", ghost_mm=ghost))
        return {**s, "t_s": t}

    one, two = frame(1.0, (0, 0), [5, 5]), frame(1.1, (20, 10), [25, 15])
    gone, back = frame(1.1, (20, 10), None), frame(1.2, (30, 10), [40, 40])
    assert joined(frames(COURSE, cam, one, two), *at(cam, [5, 5], [25, 15]))
    assert not joined(frames(COURSE, cam, one, gone, back), *at(cam, [5, 5], [40, 40]))
    # the flat view, now that the keeping moved out of it
    flat = frames(COURSE, cam, one, two, view="draw")
    assert joined(flat, [0, 0], [20, 10]) and joined(flat, [5, 5], [25, 15])
    assert not joined(frames(COURSE, cam, one, gone, back, view="draw"), [5, 5], [40, 40])


def test_the_marker_ink_lies_flat_on_the_3d_plate():
    cam, walled = scene3d.shots()["wide"], {**COURSE, "walls": [[[-60, -30], [60, -30]]]}
    drawn = scene3d.draw(walled, scene3d.snap(valid=False, race=race("idle")), cam)
    assert joined(drawn, *at(cam, [-60, -30], [60, -30]))


def test_the_next_gate_is_drawn_again_over_what_stands():
    cam = scene3d.shots()["wide"]
    first, second = at(cam, [80 + TAKE_R, 80], [80 + TAKE_R, -80])

    def rings(s, ring):
        """How many rings start at `ring`, the east point of a gate's take ring."""
        starts = [call[1:3] for call in scene3d.draw(COURSE, s, cam) if call[0] == "moveTo"]
        return sum(math.dist(ring, p) < 0.01 for p in starts)

    assert rings(scene3d.snap(valid=False, race=race("idle")), first) == 1
    assert rings(scene3d.snap(valid=False, race=race("running")), first) == 2
    # one gate on it is the second gate's ring
    on = scene3d.snap(valid=False, race=race("running", reached=1))
    assert rings(on, first) == 1 and rings(on, second) == 2
    # with no course on the board there is no gate to draw, whatever the race says
    assert scene3d.draw(None, scene3d.snap(valid=False, race=race("running")), cam)
    # it is the last thing shut before the ball: after the walls, under the ball
    walled = {**COURSE, "walls": [[[-60, -30], [60, -30]]]}
    drawn = scene3d.draw(walled, scene3d.snap(race=race("running")), cam)
    ball = drawn.index(["fillStyle=", "gradient"])
    shut = max(i for i, call in enumerate(drawn[:ball]) if call[0] == "closePath")
    start = max(i for i, call in enumerate(drawn[:shut]) if call[0] == "moveTo")
    assert math.dist(drawn[start][1:3], first) < 0.01
