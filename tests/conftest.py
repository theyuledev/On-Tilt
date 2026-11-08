import importlib

import pytest

from src.io import settings

# every module level path to this mac's lens or board json
LOCAL_JSON = [
    ("src.vision.calib", "CALIB_PATH"),
    ("tools.calibrate", "CALIB_PATH"),
    ("tools.mark_board", "CALIB_PATH"),
    ("tools.mark_board", "OUT"),
    ("tools.servo_test", "BOARD"),
]


@pytest.fixture(autouse=True)
def no_local_settings(monkeypatch, tmp_path):
    # tests see the checked in numbers, not this mac's settings.local.toml
    monkeypatch.setattr(settings, "LOCAL", tmp_path / "settings.local.toml")


@pytest.fixture(autouse=True)
def no_local_json(monkeypatch, tmp_path):
    # same for the lens and board files, a test that needs one passes it in
    for name, attr in LOCAL_JSON:
        module = importlib.import_module(name)
        monkeypatch.setattr(module, attr, tmp_path / getattr(module, attr).name)
    calib = importlib.import_module("src.vision.calib")
    for fn in (calib.load_calib, calib.save_calib):
        monkeypatch.setattr(fn, "__defaults__", (calib.CALIB_PATH,))
