import shutil

import pytest

from tests.unit import scene3d
from tests.unit.page import run, snap

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

OPEN = ("open", None)
CAM = {"camera": True}
READ = [[-90, 0], [0, 40], [90, 0]]
FLAT = {"x": 0, "y": 0, "zoom": 1, "tall": False}


def click(s, config=CAM, query=""):
    steps = [OPEN, ("snap", s), ("click", "readline"), ("frame", None)]
    out = run(steps, config=config, query=query)
    return out["page"], [m for m in out["sent"] if "line" in m]


def path_mode(**over):
    return snap(choices={**snap()["choices"], "mode": "path"}, **over)


def test_read_line_is_on_while_disarmed_with_a_camera_and_one_click_asks_once():
    page, asked = click(snap())
    assert page["readline"]["text"] == "READ LINE" and not page["readline"]["disabled"]
    assert page["linesay"]["hidden"]
    assert asked == [{"x": 0, "y": 0, "line": True}]


@pytest.mark.parametrize(
    "s, config, query",
    [
        (snap(armed=True), CAM, ""),
        (snap(), {"camera": False}, ""),
        (snap(camera_lost=True), CAM, ""),
        (snap(), CAM, "?watch"),
    ],
)
def test_read_line_is_off_while_armed_without_a_camera_or_on_a_watch_tab(s, config, query):
    page, asked = click(s, config, query)
    assert page["readline"]["disabled"] and asked == []


def test_reading_shows_on_the_button_and_a_second_click_asks_nothing():
    page, asked = click(snap(line="reading"))
    assert page["readline"]["text"] == "READING..." and page["readline"]["disabled"]
    assert page["linesay"]["hidden"] and asked == []


def test_a_refusal_shows_the_readers_sentence_and_a_good_read_asks_for_a_look():
    page, _ = click(snap(line="more than one line"))
    assert page["linesay"]["text"] == "more than one line" and not page["linesay"]["hidden"]
    assert not page["readline"]["disabled"]
    page, _ = click(snap(line="ok"))
    assert page["linesay"]["text"] == "line read, check the path on the board before arming"
    assert not page["linesay"]["hidden"]


def test_the_banner_says_finished_once_the_ball_is_at_the_end():
    shown = run([OPEN, ("snap", path_mode(armed=True, path={"done": True, "points": READ}))])
    assert shown["page"]["banner"]["text"] == "finished"
    going = run([OPEN, ("snap", path_mode(armed=True, path={"done": False, "points": READ}))])
    assert going["page"]["banner"]["text"] == "the plate rolls the ball round the path"


def test_the_ribbon_is_the_path_being_followed_before_the_one_the_page_loaded_with():
    s = {**scene3d.snap(mode="path"), "path": {"done": False, "points": READ}}
    no_path = {**scene3d.snap(mode="path"), "path": None}
    assert scene3d.run(
        ("shownPath", [None, scene3d.CFG, s]), ("shownPath", [None, scene3d.CFG, no_path])
    ) == [READ, scene3d.OWN_PATH]
    drawn = scene3d.run(("draw", ["PEN", scene3d.CFG, None, s, FLAT]))[0]
    assert all(scene3d.joined(drawn, a, b) for a, b in zip(READ, READ[1:], strict=False))
