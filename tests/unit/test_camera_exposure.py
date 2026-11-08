import shutil
from types import SimpleNamespace

import pytest

from src.io.camera_exposure import exposure_problem, uvc_target

pytestmark = pytest.mark.unit

UVC = "0x1234:0x5678"
RIG = {"camera_exposure_ms": 3.0, "camera_uvc": UVC}
HEAD = """------------ -------------- ------------ ------------ ----------------
Index        Vend:Prod      LocationId   UVC Version  Device name
------------ -------------- ------------ ------------ ----------------
"""
ROW0 = "           0 0x046d:0x082d  0x14200000         1.00   HD Pro Webcam C920\n"
ROW1 = "           1 0x1234:0x5678  0x14100000         1.00   USB Camera\n"
KW = {"capture_output": True, "text": True}


def fake_run(
    exposure="exposure-time-abs = 30", mode="auto-exposure-mode = 1", code=0, err="", listing=""
):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        out = exposure if cmd[-1] == "exposure-time-abs" else mode
        if cmd[-1] == "-d":
            out = listing
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
    assert run.calls == [
        (["/x/uvc-util", "-V", UVC, "-g", "exposure-time-abs"], KW),
        (["/x/uvc-util", "-V", UVC, "-g", "auto-exposure-mode"], KW),
    ]


def test_camera_uvc_selects_by_vendor_product_without_listing():
    run = fake_run()
    assert uvc_target(RIG, "/x/uvc-util", run) == ["-V", UVC]
    assert run.calls == []


def test_one_listed_camera_is_picked_by_its_index():
    run = fake_run(listing=HEAD + ROW0)
    assert exposure_problem({"camera_exposure_ms": 3.0}, run) is None
    assert [c for c, _ in run.calls] == [
        ["/x/uvc-util", "-d"],
        ["/x/uvc-util", "-I", "0", "-g", "exposure-time-abs"],
        ["/x/uvc-util", "-I", "0", "-g", "auto-exposure-mode"],
    ]
    assert run.calls[0][1] == KW


def test_two_listed_cameras_without_camera_uvc_are_unverified():
    run = fake_run(listing=HEAD + ROW0 + ROW1)
    assert exposure_problem({"camera_exposure_ms": 3.0}, run) == (
        "exposure unverified: several uvc cameras plugged in, "
        "set camera_uvc in configs/settings.local.toml"
    )
    assert [c for c, _ in run.calls] == [["/x/uvc-util", "-d"]]


def test_no_listed_camera_is_unverified():
    msg = exposure_problem({"camera_exposure_ms": 3.0}, fake_run(listing=HEAD))
    assert msg == "exposure unverified: no uvc camera listed"
