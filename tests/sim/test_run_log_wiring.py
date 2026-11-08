import asyncio
import json
import time

import pytest

from src import app
from src.io.run_log import start_run_log

pytestmark = pytest.mark.sim


class Server:
    def __init__(self, config):
        pass

    async def serve(self):
        # one turn of the event loop is enough for the ticker to run
        await asyncio.sleep(0)


async def one_tick(loop):
    loop.tick(1.0, 0.01)


def off(*a):
    raise OSError("off")


def no_camera(*a):
    raise RuntimeError("no camera")


@pytest.fixture
def headless(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "HandInput", off)
    monkeypatch.setattr(app, "open_camera", no_camera)
    monkeypatch.setattr(app.uvicorn, "Server", Server)
    monkeypatch.setattr(app, "run", one_tick)
    monkeypatch.setattr(start_run_log, "__defaults__", (tmp_path, time.localtime))


def test_run_log_1_writes_each_tick_from_main(headless, tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_LOG", "1")
    app.main()
    (path,) = (tmp_path / "data" / "runs").glob("*.jsonl")
    (line,) = path.read_text().splitlines()
    assert json.loads(line)["t_s"] == 1.0


def test_main_writes_nothing_unless_run_log_is_1(headless, tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_LOG", "0")
    app.main()
    assert not (tmp_path / "data").exists()
