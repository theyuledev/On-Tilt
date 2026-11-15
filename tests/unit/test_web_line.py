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
from src.vision.line_path import MANY, LinePath
from src.vision.maze_path import length
from src.vision.pipeline import Pipeline
from src.vision.tracker import BallPx
from src.web.line_read import LINE_READS, SAY, read_line, use_line
from src.web.server import make_app
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
    """A sim loop whose camera has the next of its pictures each time the frame is looked at,
    and the ball the tracker saw in it when balls are given."""

    def __init__(self, pictures, gate=None, arm_at=None, cfg=None, balls=None):
        self.pictures, self.looks = itertools.cycle(pictures), 0
        self.gate, self.arm_at, self.balls = gate, arm_at, balls and itertools.cycle(balls)
        super().__init__(cfg or load())

    @property
    def frame(self):
        if self.gate is not None:
            self.gate.wait(10)
        self.looks += 1
        if self.looks == self.arm_at:
            self.safety.arm()
        if self.balls:
            self.ball_px = next(self.balls)
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
        ([picture(VIS, TWO)], SAY[MANY]),
        ([picture(VIS, CIRCLE)], "line branches or crosses itself"),
        ([picture(VIS, [[(0.0, 0.0), (25.0, 0.0)]])], "no line found"),
        ([picture(VIS, [s]) for s in SPLIT], "pictures do not agree"),
    ],
)
def test_each_refusal_reaches_the_snapshot_and_changes_nothing(pictures, reason):
    loop = Board(pictures)
    mode, path = loop.mode, loop.cfg["control"]["path_mm"]
    assert read(loop)["line"] == reason
    assert loop.mode == loop.cfg["control"]["mode"] == mode
    assert loop.cfg["control"]["path_mm"] == path


def test_two_lines_need_the_ball_on_an_end_of_one_and_then_that_one_is_the_path():
    end = TWO[1][0]
    loop = Board([picture(VIS, TWO, 4)])
    path = loop.cfg["control"]["path_mm"]
    said = read(loop)["line"]
    assert said == "more than one line: put the ball on an end of the line you want"
    assert (loop.mode, loop.cfg["control"]["path_mm"]) == (load()["control"]["mode"], path)
    on_end = Board([picture(VIS, TWO, 4, ball=(end, 8.0))])
    on_end.ball_px = ball_px(VIS, end, 8.0)
    assert read(on_end)["line"] == "ok" and on_end.mode == "path"
    assert along(on_end.cfg["control"]["path_mm"], TWO[1])[0].max() < 2


def test_each_picture_is_read_with_the_ball_seen_at_that_moment():
    end, lost = TWO[1][0], BallPx(0.0, 0.0, 0.0, False, 0.0)
    seen = ball_px(VIS, end, 8.0)
    # two pictures with the ball lost: paired the wrong way round only two would see it
    balls = [lost, lost, seen, seen, seen]
    loop = Board([picture(VIS, TWO, 4, ball=(end, 8.0))], balls=balls)
    assert read(loop)["line"] == "ok"
    assert along(loop.cfg["control"]["path_mm"], TWO[1])[0].max() < 2


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
    loop, got = Loop(load()), []
    loop.frame = Frame(picture(VIS, [S_CURVE], 4), time.monotonic())
    reader = threading.Thread(target=lambda: got.append(read_line(loop, Pipeline(loop.cfg))))
    reader.daemon = True
    reader.start()
    reader.join(2)
    assert got == [LinePath([], "no camera picture")]


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
