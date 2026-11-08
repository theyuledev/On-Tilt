import shutil
from types import SimpleNamespace

import pytest

from src.io.camera_exposure import exposure_problem

pytestmark = pytest.mark.unit

RIG = {"camera_exposure_ms": 3.0}


def fake_run(exposure="exposure-time-abs = 30", mode="auto-exposure-mode = 1", code=0, err=""):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        out = exposure if cmd[-1] == "exposure-time-abs" else mode
        return SimpleNamespace(returncode=code, stdout=out, stderr=err)

    run.calls = calls
    return run


@pytest.fixture(autouse=True)
def tool(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/x/uvc-util")


def test_matching_values_are_fine():
    assert exposure_problem(RIG, fake_run()) is None


def test_wrong_exposure():
    msg = exposure_problem(RIG, fake_run(exposure="exposure-time-abs = 120"))
    assert "120" in msg and "30" in msg and "mode" not in msg


def test_wrong_mode():
    msg = exposure_problem(RIG, fake_run(mode="auto-exposure-mode = 8"))
    assert "8" in msg and "wanted 1" in msg


def test_both_wrong():
    run = fake_run(exposure="exposure-time-abs = 120", mode="auto-exposure-mode = 8")
    msg = exposure_problem(RIG, run)
    assert "120" in msg and "; " in msg and "mode 8" in msg


def test_failed_read_is_unverified_with_tail():
    msg = exposure_problem(RIG, fake_run(exposure="no device", code=1))
    assert msg.startswith("exposure unverified: exposure-time-abs: ")
    assert "no device" in msg


def test_number_without_equals_uses_last_integer():
    assert exposure_problem(RIG, fake_run(exposure="exposure-time-abs: 30")) is None


def test_no_tool(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert exposure_problem(RIG, fake_run()) == "exposure unverified: uvc-util not on PATH"


def test_commands_sent():
    run = fake_run()
    exposure_problem(RIG, run)
    kw = {"capture_output": True, "text": True}
    assert run.calls == [
        (["/x/uvc-util", "-I", "0", "-g", "exposure-time-abs"], kw),
        (["/x/uvc-util", "-I", "0", "-g", "auto-exposure-mode"], kw),
    ]
