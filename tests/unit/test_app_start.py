import asyncio

import pytest

from src import app

pytestmark = pytest.mark.unit

MISMATCH = "exposure 120 (100 us steps), wanted 30"


class Server:
    def __init__(self, config):
        pass

    async def serve(self):
        await asyncio.sleep(0)


class Cam:
    running = True

    def __init__(self, capture, process=None):
        self.process = process

    def read(self):
        return None

    def latest(self):
        return None

    def close(self):
        pass


def boom(*a):
    raise OSError("uvc-util: no such file")


def start(monkeypatch, open_camera, exposure_problem):
    loops = []

    async def keep(loop):
        loops.append(loop)

    monkeypatch.setattr(app, "HandInput", boom)
    monkeypatch.setattr(app, "open_camera", open_camera)
    monkeypatch.setattr(app, "CameraSource", Cam)
    monkeypatch.setattr(app, "exposure_problem", exposure_problem)
    monkeypatch.setattr(app.uvicorn, "Server", Server)
    monkeypatch.setattr(app, "run", keep)
    monkeypatch.setenv("RUN_LOG", "0")
    app.main()
    (loop,) = loops
    loop.mode = "balance"
    return loop


def test_an_os_error_opening_the_camera_starts_with_the_camera_off(monkeypatch, capsys):
    loop = start(monkeypatch, boom, lambda rig: MISMATCH)
    assert loop.frames is None and loop.exposure is None and loop.arm_blocked is None
    assert "camera off: uvc-util" in capsys.readouterr().out


def test_an_os_error_reading_the_exposure_keeps_the_camera_and_never_blocks(monkeypatch):
    loop = start(monkeypatch, lambda rig: object(), boom)
    assert isinstance(loop.frames, Cam) and loop.frames.process is not None
    assert loop.exposure == "exposure unverified: uvc-util: no such file"
    assert loop.arm_blocked is None


def test_a_confirmed_mismatch_reaches_the_loop(monkeypatch):
    loop = start(monkeypatch, lambda rig: object(), lambda rig: MISMATCH)
    assert loop.exposure == MISMATCH and loop.arm_blocked == MISMATCH
