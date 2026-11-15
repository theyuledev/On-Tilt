import itertools
import math
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.io.ports import Frame
from src.io.settings import load
from src.loop import Loop
from src.vision.line_path import LinePath
from src.vision.maze_path import length
from src.vision.pipeline import Pipeline
from src.web.server import LINE_READS, make_app, use_line
from tests.unit.line_pictures import (
    HALF,
    S_CURVE,
    STRAIGHT,
    TURNED,
    VIS,
    along,
    ball_px,
    picture,
)

pytestmark = pytest.mark.unit

TWO = [[(-80.0, -30.0), (80.0, -30.0)], [(-80.0, 30.0), (80.0, 30.0)]]
CIRCLE = [[(50 * math.cos(a), 50 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 73)]]
OTHER = [(-90.0, 60.0), (80.0, 60.0)]
# two reads of one line, two of another and an empty board
SPLIT = [STRAIGHT, OTHER, STRAIGHT, OTHER, []]


class Board(Loop):
    """A sim loop whose camera has the next of its pictures each time the frame is looked at."""

    def __init__(self, pictures, gate=None, arm_at=None, cfg=None):
        self.pictures, self.looks = itertools.cycle(pictures), 0
        self.gate, self.arm_at = gate, arm_at
        super().__init__(cfg or load())

    @property
    def frame(self):
        if self.gate is not None:
            self.gate.wait(10)
        self.looks += 1
        if self.looks == self.arm_at:
            self.safety.arm()
        return Frame(next(self.pictures), time.monotonic())

    @frame.setter
    def frame(self, _):
        pass


def ask(ws, msg=None):
    ws.send_json(msg or {})
    return ws.receive_json()


def read(loop, pipe=...):
    """Asks for one read over the socket and waits for its answer in the snapshot."""
    app = make_app(loop)
    app.state.pipe = Pipeline(loop.cfg) if pipe is ... else pipe
    with TestClient(app).websocket_connect("/ws") as ws:
        snap, end = ask(ws, {"line": True}), time.monotonic() + 30
        while snap.get("line") == "reading" and time.monotonic() < end:
            time.sleep(0.01)
            snap = ask(ws)
    return snap


def test_a_drawn_line_becomes_the_path_the_path_mode_follows():
    loop = Board([picture(VIS, [S_CURVE], 4)])
    before = loop.mode
    snap = read(loop)
    assert before != "path" and snap["line"] == "ok"
    assert loop.mode == loop.cfg["control"]["mode"] == "path"
    assert snap["choices"]["mode"] == "path"
    assert along(loop.cfg["control"]["path_mm"], S_CURVE)[0].max() < 2
    assert along(snap["path"]["points"], S_CURVE)[0].max() < 2
    assert loop.looks == LINE_READS
    # the disarmed tick builds the mode again from the new path
    loop.tick(1.0, 0.01)
    shown = loop.snapshot()["path"]["points"]
    assert shown in (snap["path"]["points"], snap["path"]["points"][::-1])


def test_the_ball_on_one_end_is_not_read_as_ink_and_the_path_starts_under_it():
    end = S_CURVE[0]
    loop = Board([picture(VIS, [S_CURVE], 4, ball=(end, 8.0))])
    loop.ball_px = ball_px(VIS, end, 8.0)
    assert read(loop)["line"] == "ok"
    path = loop.cfg["control"]["path_mm"]
    assert min(math.dist(path[0], end), math.dist(path[-1], end)) < 5
    assert along(path, S_CURVE)[0].max() < 2


def test_one_picture_of_five_cut_short_does_not_change_the_path():
    full, short = picture(VIS, [S_CURVE], 4), picture(VIS, [S_CURVE[:20]], 4)
    loop = Board([short, full, full, full, full])
    assert read(loop)["line"] == "ok"
    path = loop.cfg["control"]["path_mm"]
    off, arc = along(path, S_CURVE)
    assert off.max() < 2 and arc.max() - arc.min() > 0.9 * length(S_CURVE)


def test_the_line_is_read_on_the_board_map_the_tracker_has_now_not_the_settings_one():
    line = [(-90.0, -20.0), (HALF + 10, 30.0)]
    loop = Board([picture(TURNED, [line], 4)])
    assert read(loop, pipe=SimpleNamespace(cfg=TURNED, half_mm=HALF))["line"] == "ok"
    path = loop.cfg["control"]["path_mm"]
    assert along(path, line)[0].max() < 2
    # read up to line_edge_mm inside the board, where the line runs off it
    assert max(x for x, _ in path) == pytest.approx(HALF - TURNED["line_edge_mm"], abs=1.5)


@pytest.mark.parametrize(
    "pictures, reason",
    [
        ([picture(VIS, [])], "no line found"),
        ([picture(VIS, TWO)], "more than one line"),
        ([picture(VIS, CIRCLE)], "line branches or crosses itself"),
        ([picture(VIS, [[(0.0, 0.0), (25.0, 0.0)]])], "line too short"),
        ([picture(VIS, [s]) for s in SPLIT], "pictures do not agree"),
    ],
)
def test_each_refusal_reaches_the_snapshot_and_changes_nothing(pictures, reason):
    loop = Board(pictures)
    mode, path = loop.mode, loop.cfg["control"]["path_mm"]
    assert read(loop)["line"] == reason
    assert loop.mode == loop.cfg["control"]["mode"] == mode
    assert loop.cfg["control"]["path_mm"] == path


def test_no_camera_picture_a_lost_camera_or_no_board_map_is_refused():
    loop = Loop(load())
    assert read(loop)["line"] == "no camera picture"
    lost = Board([picture(VIS, [S_CURVE], 4)])
    lost.frames = SimpleNamespace(running=False)
    assert read(lost)["line"] == "no camera picture"
    no_map = Board([picture(VIS, [S_CURVE], 4)])
    assert read(no_map, pipe=None)["line"] == "no camera picture"
    for x in (loop, lost, no_map):
        assert x.mode == load()["control"]["mode"]
        assert x.cfg["control"]["path_mm"] == load()["control"]["path_mm"]


def test_a_camera_that_stops_sending_is_refused_not_waited_on():
    loop = Loop(load())
    loop.frame = Frame(picture(VIS, [S_CURVE], 4), time.monotonic())
    start = time.monotonic()
    assert read(loop)["line"] == "no camera picture"
    assert time.monotonic() - start < 2
    assert loop.cfg["control"]["path_mm"] == load()["control"]["path_mm"]


def test_armed_before_or_during_the_read_is_refused():
    armed = Board([picture(VIS, [S_CURVE], 4)])
    armed.safety.arm()
    assert read(armed)["line"] == "armed, kill first" and armed.looks <= 1
    midway = Board([picture(VIS, [S_CURVE], 4)], arm_at=3)
    assert read(midway)["line"] == "armed, kill first" and midway.looks == 3
    for x in (armed, midway):
        assert x.cfg["control"]["path_mm"] == load()["control"]["path_mm"]


def test_a_path_read_just_as_the_plate_arms_is_not_used():
    loop = Loop(load())
    loop.safety.arm()
    path, mode = loop.cfg["control"]["path_mm"], loop.mode
    assert use_line(loop, LinePath([[0.0, 0.0], [50.0, 0.0]], "")) == "armed, kill first"
    assert (loop.cfg["control"]["path_mm"], loop.mode) == (path, mode)


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
