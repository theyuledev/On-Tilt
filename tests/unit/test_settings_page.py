import tomllib

import pytest
from fastapi.testclient import TestClient

from src.io import settings
from src.io.settings import load, save_local, toml_value
from src.loop import Loop
from src.modes import Balance
from src.web.form import LABEL
from src.web.server import make_app
from tests.unit.test_switches import Hand

pytestmark = pytest.mark.unit

URL = "/settings.json"


@pytest.fixture
def loop():
    return Loop(load(), Hand())


@pytest.fixture
def page(loop):
    return TestClient(make_app(loop))


def local() -> dict:
    return tomllib.loads(settings.LOCAL.read_text()) if settings.LOCAL.exists() else {}


def test_the_page_gets_every_setting_by_section_and_what_locks_them(page, loop):
    got = page.get(URL).json()
    assert got["sections"] == load() and list(got["sections"]) == list(load())
    assert (got["armed"], got["locked"], got["mazes"]) == (False, ["rig"], ["square"])
    assert got["choices"]["steer"] == "hand" and got["target"] == "centre"
    kinds = {p: type(load()[p.split(".")[0]][p.split(".")[1]]) for p in got["floats"]}
    assert set(kinds.values()) == {float} and "rig.camera_exposure_ms" in kinds
    assert not {"rig.center_x_us", "rig.actuator", "vision.plate_centre_px"} & set(kinds)
    loop.arm()
    assert page.get(URL).json()["armed"] is True
    assert "/static/settings.js" in page.get("/settings").text


def test_a_change_goes_to_the_local_file_with_a_label_and_runs_at_once(page, loop):
    base = settings.BASE.read_bytes()
    got = page.post(URL, json={"control.kp_deg_mm": 0.06, "vision.plate_centre_px": [600, 380]})
    assert got.status_code == 200 and got.json()["sections"]["control"]["kp_deg_mm"] == 0.06
    assert local() == {
        "control": {"kp_deg_mm": {"value": 0.06, "label": LABEL}},
        "vision": {"plate_centre_px": {"value": [600, 380], "label": LABEL}},
    }
    assert loop.cfg["control"]["kp_deg_mm"] == 0.06 and load()["control"]["kp_deg_mm"] == 0.06
    assert loop.cfg["vision"]["plate_centre_px"] == [600, 380]
    assert settings.BASE.read_bytes() == base


def test_plain_settings_stay_plain_and_a_number_keeps_its_kind(page, loop):
    changes = {"game.max_race_s": 45, "game.countdown_s": 5, "input.hand_flip_x": True}
    assert page.post(URL, json=changes).status_code == 200
    assert local()["input"] == {"hand_flip_x": True}
    assert local()["game"]["max_race_s"]["value"] == 45
    assert type(loop.cfg["game"]["max_race_s"]) is int
    assert loop.cfg["game"]["countdown_s"] == 5.0 and type(loop.cfg["game"]["countdown_s"]) is float


def test_an_armed_loop_takes_no_settings(page, loop):
    loop.arm()
    got = page.post(URL, json={"control.kp_deg_mm": 0.06})
    assert (got.status_code, got.json()) == (409, {"error": "armed, kill first"})
    assert not settings.LOCAL.exists() and loop.cfg["control"]["kp_deg_mm"] == 0.05
    loop.kill()
    assert page.post(URL, json={"control.kp_deg_mm": 0.06}).status_code == 409


@pytest.mark.parametrize(
    ("path", "value", "why"),
    [
        ("rig.limit_x_us", 400, "read only"),
        ("rig.kill_ramp_us_s", 9000, "read only"),
        ("rig.actuator", "real", "read only"),
        ("rig.startup_flag", False, "read only"),
        ("control.nope", 1, "not a setting"),
        ("control.target", {"mm": [0, 0]}, "not a setting"),
        ("nope", 1, "not a setting"),
        ("control.kp_deg_mm", "0.06", "wants a number"),
        ("control.kp_deg_mm", True, "wants a number"),
        ("control.kp_deg_mm", None, "wants a number"),
        ("game.max_race_s", 45.5, "whole number"),
        ("input.hand_flip_x", 1, "wants bool"),
        ("control.mode", 3, "wants str"),
        ("vision.plate_centre_px", [600], "list of 2"),
        ("vision.plate_centre_px", [600, "a"], "wants a number"),
        ("vision.px_per_mm", 0, "above 0"),
        ("control.mode", "nope", "one of"),
        ("input.hand_mode", "fist", "one of"),
        ("game.maze", "no-such-maze", "does not load"),
    ],
)
def test_a_bad_change_is_refused_whole_and_nothing_is_written(page, loop, path, value, why):
    before = load()
    got = page.post(URL, json={"control.kd_deg_s_mm": 0.04, path: value})
    assert got.status_code == 400 and why in got.json()["error"]
    assert not settings.LOCAL.exists() and loop.cfg == before


def test_saved_mode_and_hand_mode_run_at_once_and_a_mode_that_cant_build_is_refused(page, loop):
    assert (
        page.post(URL, json={"control.mode": "balance", "input.hand_mode": "palm"}).status_code
        == 200
    )
    assert loop.mode == "balance" and isinstance(loop.step, Balance) and loop.hand.mode == "palm"
    assert local()["control"] == {"mode": "balance"} and local()["input"] == {"hand_mode": "palm"}
    got = page.post(URL, json={"control.mode": "path", "control.path_mm": []})
    assert got.status_code == 400 and loop.mode == "balance"
    assert page.post(URL, json={"game.maze": "square"}).status_code == 200


def test_the_local_file_keeps_its_other_lines(page):
    settings.LOCAL.write_text(
        '# rig only bits\n[rig]\nactuator = "sim"  # bench today\nserial_port = "/dev/cu.x"\n\n'
        '[control]\nkp_deg_mm = { value = 0.07, label = "player-tuned (lucas)" }\n'
    )
    assert page.post(URL, json={"control.kp_deg_mm": 0.06, "input.expo": 0.2}).status_code == 200
    assert settings.LOCAL.read_text() == (
        '# rig only bits\n[rig]\nactuator = "sim"  # bench today\nserial_port = "/dev/cu.x"\n\n'
        f'[control]\nkp_deg_mm = {{ value = 0.06, label = "{LABEL}" }}\n\n'
        f'[input]\nexpo = {{ value = 0.2, label = "{LABEL}" }}\n'
    )


def test_a_value_spread_over_lines_is_left_for_a_hand_edit(page, loop):
    text = '[vision]\nplate_centre_px = { value = [\n  640,\n  400,\n], label = "measured (x)" }\n'
    settings.LOCAL.write_text(text)
    got = page.post(URL, json={"vision.plate_centre_px": [600, 380]})
    assert got.status_code == 400 and settings.LOCAL.read_text() == text
    assert loop.cfg["vision"]["plate_centre_px"] == [640, 400]


def test_save_local_adds_to_the_end_of_a_section_and_writes_toml_values(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text('[game]\nmaze = "square"\n\n[rig]\nactuator = "sim"\n')
    save_local({("game", "ghost"): "data/runs/a.jsonl", ("game", "lost_s"): 2.0}, "by hand", path)
    assert path.read_text() == (
        '[game]\nmaze = "square"\nghost = "data/runs/a.jsonl"\n'
        'lost_s = { value = 2.0, label = "by hand" }\n\n[rig]\nactuator = "sim"\n'
    )
    assert [toml_value(v) for v in (True, 3, 0.5, 'a"b', [1, [2.0, False]])] == [
        "true",
        "3",
        "0.5",
        '"a\\"b"',
        "[1, [2.0, false]]",
    ]


def test_restart_asks_the_app_to_leave_but_never_while_armed(page, loop):
    left = []
    page.app.state.leave = lambda: left.append(True)
    loop.arm()
    got = page.post("/restart")
    assert (got.status_code, got.json()) == (409, {"error": "armed, kill first"})
    assert not left and not page.app.state.restart
    loop.safety = type(loop.safety)()
    assert page.post("/restart").json() == {"restarting": True}
    assert left == [True] and page.app.state.restart
