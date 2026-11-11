import math
from pathlib import Path

import pytest

from src.contracts import Intent
from src.io.camera import ClipSource
from src.io.settings import load
from src.loop import Loop
from src.vision.pipeline import Pipeline
from tests.replay.test_guard_clip import CLIP
from tests.sim.test_edge_guard import reversals

pytestmark = pytest.mark.replay

START, FRAMES = 13000, 460


class Pusher:
    """Full stick toward the rim nearest the ball the loop last saw."""

    seen = calibrating = False
    progress = 0.0

    def __init__(self):
        self.intent = Intent(0.0, 0.0, valid=False, t_s=0.0)

    def latest(self) -> Intent:
        return self.intent

    def recentre(self) -> None:
        pass

    def push(self, ball, now: float) -> None:
        if ball is None or not ball.valid:
            self.intent = Intent(0.0, 0.0, valid=True, t_s=now)
            return
        r = math.hypot(ball.x_mm, ball.y_mm) or 1.0
        self.intent = Intent(ball.x_mm / r, ball.y_mm / r, valid=True, t_s=now)


def rate(guard_on: bool, stamps: list[float]) -> dict:
    cfg = load(local=Path("/dev/null"))
    # full stick is the 2 deg cap whichever way the shaping sets its scale
    cfg["control"].update(mode="hand", guard_on=guard_on, player_max_deg=2.0)
    hand = Pusher()
    frames = ClipSource(CLIP, START, process=Pipeline(cfg).step)
    loop = Loop(cfg, hand=hand, frames=frames)
    loop.arm()
    pulses, prev = [], stamps[0]
    for t in stamps:
        hand.push(loop.ball, t)
        rec = loop.tick(t, t - prev)
        pulses.append((rec.servo_cmd.x_us, rec.servo_cmd.y_us))
        prev = t
    frames.close()
    band = cfg["control"]["servo_dead_band_us"]
    return {a: reversals([p[i] for p in pulses], band) for i, a in enumerate("xy")}


@pytest.fixture(scope="module")
def rates():
    if not CLIP.exists():
        pytest.skip(f"{CLIP.name} not on this machine")
    stamps = [float(t) for t in CLIP.with_suffix(".csv").read_text().split()]
    stamps = stamps[START : START + FRAMES]
    return rate(True, stamps), rate(False, stamps)


@pytest.mark.parametrize("axis", "xy")
def test_the_guard_does_not_add_chatter_on_tracker_noise(rates, axis):
    # over the 3.8 s clip: x 7 on / 7 off and y 15 / 6 before the side was held, now 9 / 7 and 6 / 6
    on, off = rates
    assert on[axis] <= off[axis] + 2
