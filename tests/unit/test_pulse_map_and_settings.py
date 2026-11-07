from pathlib import Path

import pytest

from src.contracts import PlateCommand
from src.control.pulse_map import pulse_to_tilt, tilt_to_pulse
from src.io.settings import load, load_raw

pytestmark = pytest.mark.unit

UNTESTED = "uncalibrated default (NOT TESTED, expected predicted default value)"
DATED = ("measured (", "sim-tuned (PR #", "player-tuned (")


def label_ok(label: str) -> bool:
    return label == UNTESTED or (label.startswith(DATED) and label.endswith(")"))


def test_pulse_map_round_trip():
    rig = load()["rig"]
    for tx, ty in [(0, 0), (12, -12), (-3.3, 7.7), (5, 0)]:
        back = pulse_to_tilt(tilt_to_pulse(PlateCommand(tx, ty), rig), rig)
        assert back.tilt_x_deg == pytest.approx(tx, abs=1 / rig["us_per_deg_x"])
        assert back.tilt_y_deg == pytest.approx(ty, abs=1 / rig["us_per_deg_y"])


def test_positive_tilt_raises_pulse_with_positive_scale():
    rig = load()["rig"]
    cmd = tilt_to_pulse(PlateCommand(1.0, 1.0), rig)
    assert cmd.x_us > rig["center_x_us"] and cmd.y_us > rig["center_y_us"]


def test_every_number_in_settings_has_a_label():
    for name, section in load_raw().items():
        for key, v in section.items():
            if isinstance(v, bool) or isinstance(v, str):
                continue
            assert isinstance(v, dict), f"{name}.{key} has no label"
            assert label_ok(str(v.get("label", ""))), f"{name}.{key} label"


def test_label_check_wants_the_whole_untested_string():
    assert label_ok(UNTESTED)
    assert label_ok("sim-tuned (PR #32)") and label_ok("measured (2026-11-07)")
    assert not label_ok("measured")
    assert not label_ok("uncalibrated default (estimated expected value)")
    assert not label_ok("uncalibrated default")


def test_tests_ignore_the_local_settings_file():
    assert load() == load(local=Path("nope.toml"))
    assert load_raw() == load_raw(local=Path("nope.toml"))
