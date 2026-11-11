import shutil

import pytest
from fastapi.testclient import TestClient

from src.io.settings import load
from src.loop import Loop
from src.web.server import make_app
from tests.unit.page import run

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

URL = "/settings.json"


@pytest.fixture
def served():
    return TestClient(make_app(Loop(load()))).get(URL).json()


def play(served, steps=(), answer=None):
    urls = {URL: served, f"POST {URL}": answer or [200, served]}
    return run(steps, page="settings.html", urls=urls)


def fields(page) -> dict:
    """Each field the page built, by its settings path: label, unit and its input or switch."""
    out = {}
    for section in page["sections"]["kids"]:
        for label, editor, key in (f["kids"] for f in section["kids"][1]["kids"]):
            unit = editor["kids"][1]["text"] if len(editor["kids"]) > 1 else ""
            control = editor["kids"][0] if editor["cls"].startswith("value") else editor
            out[key["text"]] = {"label": label["text"], "unit": unit, "box": editor, **control}
    return out


def pressed(page, group):
    return [(k["attrs"]["aria-pressed"], k["disabled"]) for k in page[group]["kids"]]


def test_every_setting_gets_a_field_named_and_united_from_its_key(served):
    page = play(served)["page"]
    got = fields(page)
    shown_elsewhere = {"control.mode", "input.hand_mode", "game.maze", "game.ghost"}
    every = {f"{s}.{k}" for s, sec in served["sections"].items() for k in sec}
    assert set(got) == every - shown_elsewhere
    names = {path: (f["label"], f["unit"]) for path, f in got.items()}
    assert names["rig.center_x_us"] == ("CENTER X", "µs")
    assert names["rig.us_per_deg_x"] == ("US PER DEG X", "µs/°")
    assert names["rig.slew_us_s"] == ("SLEW", "µs/s")
    assert names["rig.camera_fps"] == ("CAMERA", "fps")
    assert names["control.kd_deg_s_mm"] == ("KD", "°·s/mm")
    assert names["control.sim_g_mm_s2"] == ("SIM G", "mm/s²")
    assert names["vision.px_per_mm"] == ("PX PER MM", "px/mm")
    assert names["vision.velocity_span_frames"] == ("VELOCITY SPAN FRAMES", "")
    assert names["input.hand_min_cutoff_hz"] == ("HAND MIN CUTOFF", "Hz")
    assert got["control.kp_deg_mm"]["value"] == "0.05"
    # a whole float keeps its point, a whole number and a list stay as they are
    assert served["sections"]["rig"]["camera_exposure_ms"] == 3
    assert got["rig.camera_exposure_ms"]["value"] == "3.0"
    assert got["rig.center_x_us"]["value"] == "1500"
    assert got["vision.plate_centre_px"]["value"] == "[640,400]"
    assert got["vision.plate_centre_px"]["value"] == "[640,400]"
    assert got["rig.actuator"]["value"] == "sim"
    assert got["game.attract"]["attrs"]["aria-checked"] == "false"
    titles = [k["text"] for k in page["jump"]["kids"]]
    assert titles[0] == "RIG AND SERVOS" and len(titles) == 5


def test_the_rig_is_read_only_and_save_waits_for_a_change(served):
    page = play(served)["page"]
    got = fields(page)
    assert all(f["disabled"] for path, f in got.items() if path.startswith("rig."))
    assert not any(f["disabled"] for path, f in got.items() if not path.startswith("rig."))
    assert page["save"]["disabled"]


def test_save_sends_only_what_changed_as_the_kind_of_value_it_is(served):
    steps = [
        ("type", ["control.kp_deg_mm", "0.06"]),
        ("type", ["vision.plate_centre_px", "[600, 380]"]),
        ("type", ["input.expo", " .25 "]),
        ("type", ["control.kd_deg_s_mm", "0.0375"]),
        ("click", "game.attract"),
        ("click", "pickmode/1"),
    ]
    out = play(served, steps)
    assert not out["page"]["save"]["disabled"] and out["posts"] == []
    out = play(served, [*steps, ("click", "save")])
    assert out["posts"] == [
        {
            "control.kp_deg_mm": 0.06,
            "vision.plate_centre_px": [600, 380],
            "input.expo": 0.25,
            "game.attract": True,
            "control.mode": "balance",
        }
    ]
    assert out["page"]["note"]["text"].startswith("Saved to settings.local.toml")
    assert out["page"]["save"]["disabled"]


@pytest.mark.parametrize("text", ["", "abc", "1,5", "[1]", "[1, 2"])
def test_something_that_does_not_read_as_the_value_blocks_save(served, text):
    path = "vision.plate_centre_px" if text.startswith("[") else "control.kp_deg_mm"
    out = play(served, [("type", [path, text]), ("click", "save")])
    assert fields(out["page"])[path]["box"]["cls"] == "value bad"
    assert out["page"]["save"]["disabled"] and out["posts"] == []
    back = "[640,400]" if text.startswith("[") else "0.05"
    out = play(served, [("type", [path, text]), ("type", [path, back])])
    assert fields(out["page"])[path]["box"]["cls"] == "value" and out["page"]["save"]["disabled"]


def test_how_you_play_shows_what_runs_and_only_offers_what_has_code_behind_it(served):
    page = play(served)["page"]
    assert pressed(page, "pickmode") == [("true", False), ("false", False), ("false", False)]
    assert pressed(page, "pickhand") == [("true", False), ("false", False)]
    # no drawn maze file here, and no code for "none"
    assert pressed(page, "pickmaze") == [("false", True), ("true", False), ("false", True)]
    assert pressed(page, "picksteer") == [("false", True), ("true", True)]
    assert pressed(page, "picktarget") == [("true", True), ("false", True), ("false", True)]
    assert not page["pickguard"]["disabled"]
    page = play({**served, "mazes": ["drawn", "square"]}, [("click", "pickmaze/0")])["page"]
    assert pressed(page, "pickmaze") == [("true", False), ("false", False), ("false", True)]


def test_an_armed_plate_locks_the_page(served):
    out = play(
        {**served, "armed": True}, [("type", ["control.kp_deg_mm", "0.06"]), ("click", "save")]
    )
    page = out["page"]
    assert all(f["disabled"] for f in fields(page).values()) and page["save"]["disabled"]
    assert all(off for _, off in pressed(page, "pickmode") + pressed(page, "pickhand"))
    assert page["note"]["text"].startswith("Armed") and page["note"]["cls"] == "red"
    assert out["posts"] == []


def test_a_refused_save_says_why_and_keeps_the_edit(served):
    steps = [("type", ["control.kp_deg_mm", "0.06"]), ("click", "save")]
    out = play(served, steps, answer=[400, {"error": "control.kp_deg_mm wants a number"}])
    note = out["page"]["note"]
    assert (note["text"], note["cls"]) == ("control.kp_deg_mm wants a number", "red")
    assert fields(out["page"])["control.kp_deg_mm"]["value"] == "0.06"
    assert not out["page"]["save"]["disabled"]


def test_revert_drops_the_edits(served):
    out = play(served, [("type", ["control.kp_deg_mm", "0.06"]), ("click", "revert")])
    assert fields(out["page"])["control.kp_deg_mm"]["value"] == "0.05"
    assert out["page"]["save"]["disabled"]
