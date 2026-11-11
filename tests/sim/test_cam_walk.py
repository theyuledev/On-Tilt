import math
import shutil
import time

import pytest

from src.contracts import BallState
from src.io.settings import load
from src.loop import Loop
from tests.unit.cam import frame, play, shots, steps

pytestmark = [
    pytest.mark.sim,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

HOLD_S, NEAR_MM = 1.0, 40


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def walk(clock, seconds=60.0):
    """The sim ball driven round the stock path by path mode, as the camera would be fed it:
    the corners are the gates, each one taken when the ball gets a ball radius close."""
    cfg = load()
    cfg["control"]["mode"] = "path"
    path = cfg["control"]["path_mm"]
    loop, dt, gates, t0 = Loop(cfg), 1 / cfg["control"]["sim_tick_hz"], list(path[1:]), clock[0]
    ring = cfg["rig"]["ball_d_mm"] / 2
    loop.ball = BallState(*path[0], 0.0, 0.0, True, clock[0])
    loop.arm()
    frames = []
    for _ in range(round(seconds / dt)):
        clock[0] += dt
        loop.keys.set(0.0, 0.0)
        b = loop.tick(clock[0], dt).ball
        if gates and math.dist((b.x_mm, b.y_mm), gates[0]) < ring:
            gates.pop(0)
        t = clock[0] - t0
        phase, gate = ("run", gates[0]) if gates else ("done", None)
        frames.append(frame(t, b.x_mm, b.y_mm, b.valid, phase, gate=gate, path=path, gate_r=ring))
    return frames, cfg["rig"]["plate_inner_mm"] / 2, ring


def test_the_camera_follows_the_sim_ball_round_the_path_in_whole_steady_shots(clock):
    frames, half, ring = walk(clock)
    assert frames[-1]["phase"] == "done", "the sim ball never finished the path"
    got = play(frames, half)
    cut = shots(got)
    names = [s for s, _, _ in cut]
    assert names[0] == "wide" and names[-1] == "tower" and set(names[1:-1]) == {"chase", "gate"}
    # a gate shot for each corner, cutting in as the ball closes on it
    assert names.count("gate") == 4
    for name, start, _ in cut:
        if name == "gate":
            first = next(i for i, s in enumerate(got) if s["t"] == start)
            assert math.dist(got[first]["eye"], frames[first]["gate"]) < ring + NEAR_MM
    # every shot but the one the finish cuts short gets its second
    assert all(end - start >= HOLD_S - 0.02 for _, start, end in cut[:-2])
    # a cut is a quick swing, never a jump, and between cuts the picture is steady
    assert max(steps(got, "cx")) < 8 and max(steps(got, "cy")) < 8
    assert max(steps(got, "yaw")) < 6 and max(steps(got, "zoom")) < 0.06
    # the opening wide shot has no cut time, it was never cut to
    settled = [
        i for i, s in enumerate(got) if s["shot"] in ("chase", "gate") and s["t"] - s["shotAt"] > 1
    ]
    assert max(steps(got, "cx")[i - 1] for i in settled) < 1.5
    assert max(steps(got, "cy")[i - 1] for i in settled) < 1.5
    # and the ball never leaves it
    assert all(0 <= s["at"][0] <= 900 and 0 <= s["at"][1] <= 900 for s in got)
    assert all(abs(s["cam"]["cx"]) <= half and abs(s["cam"]["cy"]) <= half for s in got)
