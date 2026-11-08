from pathlib import Path

import pytest

from src.control.guard import Guard
from src.control.pulse_map import center
from src.io.camera import ClipSource
from src.io.settings import load
from src.loop import Loop
from src.safety import limits
from src.vision.pipeline import Pipeline
from tests.replay.test_guard_chatter import CLIP, FRAMES, START, Pusher

pytestmark = pytest.mark.replay

MOST = 0.5


def run(guard_on: bool, stamps: list[float]):
    # module fixtures run before the conftest one that hides settings.local.toml
    cfg = load(local=Path("/dev/null"))
    cfg["control"].update(mode="hand", guard_on=guard_on)
    hand = Pusher()
    frames = ClipSource(CLIP, START, process=Pipeline(cfg).step)
    loop = Loop(cfg, hand=hand, frames=frames)
    loop.arm()
    out, prev = [], stamps[0]
    for t in stamps:
        hand.push(loop.ball, t)
        rec = loop.tick(t, t - prev)
        out.append((t, rec, loop.snapshot()))
        prev = t
    frames.close()
    return cfg, out


@pytest.fixture(scope="module")
def runs():
    if not CLIP.exists():
        pytest.skip(f"{CLIP.name} not on this machine")
    stamps = [float(t) for t in CLIP.with_suffix(".csv").read_text().split()]
    stamps = stamps[START : START + FRAMES]
    cfg, a = run(True, stamps)
    _, b = run(False, stamps)
    return cfg, a, b


def tilts(rec) -> tuple[float, float]:
    return rec.plate_cmd.tilt_x_deg, rec.plate_cmd.tilt_y_deg


def capped(cap, rec) -> list[float]:
    # the unguarded player under the guard's tilt cap, which is not the guard's correction
    return [min(max(t, -c), c) for t, c in zip(tilts(rec), cap, strict=True)]


def weighted(a, axis: int) -> list[int]:
    return [k for k, (_, _, snap) in enumerate(a) if snap["guard"]["w"][axis] > 0]


def test_the_page_gets_the_guard_weight_and_side(runs):
    _, a, b = runs
    for _, _, snap in a:
        g = snap["guard"]
        assert isinstance(g, dict) and set(g) == {"w", "side"}
        assert isinstance(g["w"], list) and isinstance(g["side"], list)
        assert len(g["w"]) == len(g["side"]) == 2
        for w, side in zip(g["w"], g["side"], strict=True):
            assert 0 <= w <= 1 and side in (-1, 0, 1)
            assert (side == 0) == (w == 0)
    assert all(snap["guard"] is None for _, _, snap in b)


def test_the_edge_roll_wakes_the_guard(runs):
    _, a, _ = runs
    # seen 459 x and 376 y of 460
    assert max(len(weighted(a, i)) for i in range(2)) > 0


def test_a_weighted_axis_leans_inward_of_the_player(runs):
    cfg, a, b = runs
    cap = Guard(cfg).cap
    lean, seen = [], []
    for i in range(2):
        for k in weighted(a, i):
            side = a[k][2]["guard"]["side"][i]
            lean.append((tilts(a[k][1])[i] - capped(cap, b[k][1])[i]) * side)
            if a[k][1].ball.valid:
                seen.append(lean[-1])
    assert max(lean) <= 0
    # a lost ball fades the guard out, so only seen-ball ticks count. seen 643 of 657
    assert sum(d < 0 for d in seen) / len(seen) > MOST


def test_guarded_pulses_stay_in_limits_and_slew(runs):
    cfg, a, _ = runs
    rig = cfg["rig"]
    x_lo, x_hi, y_lo, y_hi = limits(rig)
    prev, last_t = center(rig), a[0][0]
    for t, rec, _ in a:
        assert rec.armed
        p = (rec.servo_cmd.x_us, rec.servo_cmd.y_us)
        assert x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi
        step = rig["slew_us_s"] * (t - last_t) + 1
        assert abs(p[0] - prev[0]) <= step and abs(p[1] - prev[1]) <= step
        prev, last_t = p, t
