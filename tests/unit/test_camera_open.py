import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.io.camera import open_camera, revive_camera, set_exposure

pytestmark = pytest.mark.unit

CAM = {"camera_width": 1280, "camera_height": 800, "camera_fps": 120, "camera_exposure_ms": 3.0}
UVC = "0x1234:0x5678"
RIG = {**CAM, "camera_uvc": UVC}
TOOL = "/x/uvc-util"
MODE8 = [TOOL, "-V", UVC, "-s", "auto-exposure-mode=8"]
MODE1 = [TOOL, "-V", UVC, "-s", "auto-exposure-mode=1"]
EXPOSURE = [TOOL, "-V", UVC, "-s", "auto-exposure-mode=1", "-s", "exposure-time-abs=30"]
READBACK = [[TOOL, "-V", UVC, "-g", name] for name in ("exposure-time-abs", "auto-exposure-mode")]
TWO_CAMS = """Index        Vend:Prod      LocationId   UVC Version  Device name
------------ -------------- ------------ ------------ ----------------
           0 0x046d:0x082d  0x14200000         1.00   HD Pro Webcam C920
           1 0x1234:0x5678  0x14100000         1.00   USB Camera
"""
SEVERAL = "camera: several uvc cameras plugged in, set camera_uvc in configs/settings.local.toml\n"


def fake_run(exposure="exposure-time-abs = 30", listing=""):
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        out = exposure if cmd[-1] == "exposure-time-abs" else "auto-exposure-mode = 1"
        if cmd[-1] == "-d":
            out = listing
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    return run, ran


def fake_cameras(shapes, dead_opens=0, blank_opens=0):
    opened = []
    fast = []

    class Cap:
        def __init__(self, index):
            self.shape = shapes[index] if index < len(shapes) else None
            self.sets = {}
            self.reads = 0
            self.released = False
            opened.append(self)

        def isOpened(self):
            return self.shape is not None

        def read(self):
            # in fast mode the first blank_opens opens never give a frame and
            # the next dead_opens give one frame then nothing
            self.reads += 1
            if self in fast:
                nth = fast.index(self)
                if nth < blank_opens or (nth < blank_opens + dead_opens and self.reads > 1):
                    return False, None
            return True, np.zeros(self.shape, np.uint8)

        def set(self, prop, value):
            if not self.sets:
                fast.append(self)
            self.sets[prop] = value

        def release(self):
            self.released = True

    return Cap, opened


def test_picks_the_camera_by_frame_size_and_sets_fast_mode():
    cap_cls, opened = fake_cameras([(1080, 1920, 3), (800, 1280, 3), (720, 1280, 3)])
    run, ran = fake_run()
    cap = open_camera(RIG, cap_cls, run)
    assert cap.shape == (800, 1280, 3)
    assert cap.sets == {
        cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*"MJPG"),
        cv2.CAP_PROP_FPS: 120,
    }
    assert all(not c.sets for c in opened if c is not cap)
    assert len(ran) == (3 if shutil.which("uvc-util") else 0)


def test_exposure_goes_manual_in_100us_steps(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    ran = []
    assert set_exposure(RIG, ran.append)
    assert ran == [EXPOSURE]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert not set_exposure(RIG, ran.append) and len(ran) == 1


def test_set_exposure_skips_two_uvc_cameras_quietly(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    run, ran = fake_run(listing=TWO_CAMS)
    assert not set_exposure(CAM, run)
    assert ran == [[TOOL, "-d"]]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("tool, said", [(TOOL, SEVERAL), (None, "camera: uvc-util not on PATH\n")])
def test_revive_without_a_target_says_why_once(monkeypatch, capsys, tool, said):
    monkeypatch.setattr(shutil, "which", lambda name: tool)
    run, ran = fake_run(listing=TWO_CAMS)
    assert not revive_camera(CAM, run)
    assert ran == ([[TOOL, "-d"]] if tool else [])
    assert capsys.readouterr().out == said


def test_open_with_two_uvc_cameras_says_so_once(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    cap_cls, _ = fake_cameras([(800, 1280, 3)])
    run, _ = fake_run(listing=TWO_CAMS)
    open_camera(CAM, cap_cls, run)
    out = capsys.readouterr().out.splitlines()
    lines = [s for s in out if s.startswith("camera:")]
    assert len(lines) == 1 and "several uvc cameras" in lines[0]


def test_one_uvc_camera_is_used_by_its_index(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    run, ran = fake_run(listing=TWO_CAMS.splitlines()[2])
    assert set_exposure(CAM, run)
    assert ran == [[TOOL, "-d"], [TOOL, "-I", "0", *EXPOSURE[3:]]]


def test_reopens_once_when_the_stream_dies_after_warm_up(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    cap_cls, opened = fake_cameras([(800, 1280, 3)], dead_opens=1)
    ran = []
    cap = open_camera(RIG, cap_cls, ran.append)
    fast = [c for c in opened if c.sets]
    assert len(fast) == 2 and cap is fast[1]
    assert fast[0].released and not cap.released
    assert cap.sets == fast[0].sets
    assert not ran


@pytest.mark.parametrize("tool, cmds", [(TOOL, [MODE8, MODE1, EXPOSURE, *READBACK]), (None, [])])
def test_blank_first_open_gets_an_exposure_toggle_then_a_reopen(monkeypatch, tool, cmds):
    monkeypatch.setattr(shutil, "which", lambda name: tool)
    cap_cls, opened = fake_cameras([(800, 1280, 3)], blank_opens=1)
    run, ran = fake_run()
    cap = open_camera(RIG, cap_cls, run)
    fast = [c for c in opened if c.sets]
    assert len(fast) == 2 and cap is fast[1]
    assert fast[0].released and not cap.released
    assert ran == cmds


@pytest.mark.parametrize("kind", ["blank", "dead"])
def test_a_stalled_open_is_dropped_after_three_reads(monkeypatch, kind):
    # each failed read blocks about a second on the rig, so a stalled open can't get 30 of them
    monkeypatch.setattr(shutil, "which", lambda name: None)
    cap_cls, opened = fake_cameras([(800, 1280, 3)], **{f"{kind}_opens": 1})
    open_camera(RIG, cap_cls, [].append)
    fast = [c for c in opened if c.sets]
    assert fast[0].reads == 3 and fast[0].released


def test_gives_up_after_five_opens(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    cap_cls, opened = fake_cameras([(800, 1280, 3)], blank_opens=1, dead_opens=4)
    ran = []
    with pytest.raises(RuntimeError, match="5 opens"):
        open_camera(RIG, cap_cls, ran.append)
    assert len([c for c in opened if c.sets]) == 5
    assert all(c.released for c in opened)
    assert ran == [MODE8, MODE1] * 4


def test_good_open_prints_a_wrong_exposure_readback_once(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    cap_cls, _ = fake_cameras([(800, 1280, 3)])
    run, ran = fake_run("exposure-time-abs = 120")
    cap = open_camera(RIG, cap_cls, run)
    assert cap.shape == (800, 1280, 3) and not cap.released
    assert ran == [EXPOSURE, *READBACK]
    assert capsys.readouterr().out == "camera: exposure 120 (100 us steps), wanted 30\n"


def test_good_open_with_matching_readback_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda name: TOOL)
    cap_cls, _ = fake_cameras([(800, 1280, 3)])
    run, ran = fake_run()
    open_camera(RIG, cap_cls, run)
    assert ran == [EXPOSURE, *READBACK]
    assert capsys.readouterr().out == ""


def test_no_matching_camera_fails_clearly():
    cap_cls, _ = fake_cameras([(1080, 1920, 3), (720, 1280, 3)])
    with pytest.raises(RuntimeError, match="1280x800"):
        open_camera(RIG, cap_cls, lambda cmd: None)
