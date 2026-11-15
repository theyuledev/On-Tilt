import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.io.settings import load
from src.loop import Loop
from src.vision.pipeline import Pipeline
from src.web.line_read import LINE_READS
from src.web.server import make_app
from tests.unit.line_pictures import S_CURVE, VIS, picture
from tests.unit.test_web_line import Board, ask, read

pytestmark = pytest.mark.unit


def test_a_second_click_while_reading_starts_no_second_read_and_the_socket_keeps_answering():
    gate = threading.Event()
    loop = Board([picture(VIS, [S_CURVE], 4)], gate=gate)
    app = make_app(loop)
    app.state.pipe = Pipeline(loop.cfg)
    with TestClient(app).websocket_connect("/ws") as ws:
        assert ask(ws, {"line": True})["line"] == "reading"
        assert ask(ws, {"line": True})["line"] == "reading"
        ws.send_json({"kill": True})
        assert ws.receive_json()["line"] == "reading"
        gate.set()
        end = time.monotonic() + 30
        while ask(ws).get("line") == "reading" and time.monotonic() < end:
            time.sleep(0.01)
        assert ask(ws)["line"] == "ok"
    assert loop.looks == LINE_READS


def test_a_read_that_fails_says_so_and_the_next_one_can_start():
    loop = Board([picture(VIS, [S_CURVE], 4)])
    assert read(loop, pipe=SimpleNamespace(cfg={}, half_mm=146.0))["line"] == (
        "could not read the line"
    )
    assert read(loop)["line"] == "ok"


def test_the_snapshot_has_no_line_answer_before_any_read():
    app = make_app(Loop(load()))
    with TestClient(app).websocket_connect("/ws") as ws:
        assert "line" not in ask(ws)


def test_config_says_whether_there_is_a_camera():
    loop = Loop(load())
    assert TestClient(make_app(loop)).get("/config").json()["camera"] is False
    loop.frames = SimpleNamespace(running=True)
    assert TestClient(make_app(loop)).get("/config").json()["camera"] is True
