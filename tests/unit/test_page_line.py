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


def shown(*steps):
    return run([OPEN, *steps], config=CAM)["page"]


def test_read_line_is_on_while_disarmed_and_a_click_says_reading_at_once_and_asks_once():
    page = shown(("snap", snap()))
    assert page["readline"]["text"] == "READ LINE" and not page["readline"]["disabled"]
    assert page["linesay"]["hidden"]
    page, asked = click(snap())
    assert page["readline"]["text"] == "READING..." and page["readline"]["disabled"]
    assert asked == [{"x": 0, "y": 0, "line": True}]


def test_the_click_keeps_reading_over_the_last_answer_until_the_app_has_had_time_to_start():
    old = ("snap", snap(line="no line found"))
    steps = [old, ("click", "readline"), ("frame", None), ("wait", 600), old]
    page = shown(*steps)
    assert page["readline"]["text"] == "READING..." and page["linesay"]["hidden"]
    page = shown(*steps, ("wait", 100), old)
    assert page["readline"]["text"] == "READ LINE" and not page["linesay"]["hidden"]


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


def test_a_refusal_says_no_path_and_why_and_a_good_read_says_how_much_line_it_got():
    page = shown(("snap", snap(line="more than one line")))
    assert page["linesay"]["text"] == "NO PATH: more than one line"
    assert page["linesay"]["cls"] == "red" and not page["linesay"]["hidden"]
    assert not page["readline"]["disabled"]
    got = path_mode(line="ok", path={"done": False, "points": [[0, 0], [30, 40], [30, 140]]})
    page = shown(("snap", got))
    assert page["linesay"]["text"] == "GOT IT: 15 cm of line. check the path on the board, then ARM"
    assert page["linesay"]["cls"] == "got" and not page["linesay"]["hidden"]


def test_only_an_answer_that_ends_a_read_flashes_even_when_it_says_the_same_again():
    old = ("snap", snap(line="no line found"))
    assert shown(old)["linesay"]["cls"] == "red"
    assert shown(("snap", snap(line="reading")), old)["linesay"]["cls"] == "red flash"
    again = shown(old, ("click", "readline"), old, ("wait", 700), old)["linesay"]
    assert again["cls"] == "red flash" and again["text"] == "NO PATH: no line found"


def test_the_check_the_path_hint_goes_once_armed_or_in_another_mode():
    assert click(path_mode(line="ok", armed=True))[0]["linesay"]["hidden"]
    assert click(snap(line="ok"))[0]["linesay"]["hidden"]
    assert not click(snap(line="no line found"))[0]["linesay"]["hidden"]


def test_arm_waits_while_a_line_is_read():
    assert click(snap(line="reading"))[0]["arm"]["disabled"]
    assert not click(snap(line="ok"))[0]["arm"]["disabled"]


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
