import math

import cv2
import numpy as np
import pytest

from src.io.ports import Frame
from src.io.serial_link import BoardLink
from src.io.settings import load
from src.vision.pointer import PairTracker, report
from tests.contract.fake_board import FakeBoard
from tests.unit.test_servo_test import TICK, Clock
from tools.bench import Bench
from tools.servo_test import plan, run

pytestmark = pytest.mark.unit

US_PER_DEG, DEAD_S, TAU_S, TOP_DEG_S, DEAD_BAND_US = -12.5, 0.05, 0.025, 130.0, 15
SHAFT = (840, 180)
NOISE = (np.random.default_rng(0).random((800, 1280)) * 10).astype(np.uint8)


class FakeServo:
    """Pure dead time, first-order lag, speed cap and a dead band around the settled target."""

    def __init__(self, clock):
        self.clock, self.log = clock, [(-1.0, 1500)]
        self.angle = self.t = 0.0

    def command(self, us):
        if abs(us - self.log[-1][1]) >= DEAD_BAND_US:
            self.log.append((self.clock.t, us))

    def at(self, t):
        while self.t < t:
            dt = min(0.001, t - self.t)
            us = next(u for tc, u in reversed(self.log) if tc <= self.t - DEAD_S)
            step = ((us - 1500) / US_PER_DEG - self.angle) * dt / TAU_S
            self.angle += max(-TOP_DEG_S * dt, min(TOP_DEG_S * dt, step))
            self.t += dt
        return self.angle


class ServoBoard(FakeBoard):
    def __init__(self, servo):
        super().__init__()
        self.servo = servo

    def _handle(self, parts):
        reply = super()._handle(parts)
        if reply.startswith("OK P"):
            self.servo.command(int(reply.split()[2]))
        return reply


def horn(img, shaft, deg, gap, size, dot):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    for sign in (1, -1):
        cx, cy = shaft[0] + sign * gap / 2 * c, shaft[1] + sign * gap / 2 * s
        h = size / 2
        corners = [
            (cx + c * x - s * y, cy + s * x + c * y)
            for x, y in ((-h, -h), (h, -h), (h, h), (-h, h))
        ]
        cv2.fillConvexPoly(img, np.round(corners).astype(np.int32), 240)
        # sub pixel centre, a real camera never snaps the dot to a whole pixel
        cv2.circle(img, (round(16 * cx), round(16 * cy)), 16 * (dot // 2), 30, -1, shift=4)


class ServoCam:
    """Renders the horn notes, a still second servo, a shifting breadboard and a blinking led."""

    def __init__(self, servo, clock, horn0, gap, size, dot):
        self.servo, self.clock, self.reads = servo, clock, 0
        self.horn0, self.gap, self.size, self.dot = horn0, gap, size, dot

    def read(self):
        self.clock.t += 1 / 120
        t, self.reads = self.clock.t, self.reads + 1
        img = 40 + NOISE
        horn(img, SHAFT, self.horn0 + self.servo.at(t), self.gap, self.size, self.dot)
        horn(img, (SHAFT[0], SHAFT[1] + 350), self.horn0, self.gap, self.size, self.dot)
        x = 60 + 3 * (int(t / 0.5) % 2)
        img[250:550, x : x + 300] = 230
        if int(t * 4) % 2 == 0:
            img[700:706, 1100:1106] = 255
        return Frame(img, t)


@pytest.mark.parametrize("horn0,gap,size,dot", [(40, 88, 60, 16), (85, 60, 40, 10)])
def test_bench_recovers_the_fake_servo_parameters(horn0, gap, size, dot):
    rig = load()["rig"]
    clock = Clock(0.0002)
    servo = FakeServo(clock)
    b = Bench(BoardLink(ServoBoard(servo)), rig)
    b.submit("l")
    b.submit("A")
    steps = plan(rig, 0, TICK)
    tracker = PairTracker(max(us for us, _ in steps))
    cam = ServoCam(servo, clock, horn0, gap, size, dot)
    commands, rows = run(b, cam, steps, 1500, 0, TICK, tracker, clock)
    assert len(rows) > 0.9 * cam.reads
    p1, p2 = tracker.pair
    assert math.dist(p1, p2) == pytest.approx(gap, abs=3)
    assert np.allclose((np.array(p1) + p2) / 2, SHAFT, atol=3)
    out = report(commands, rows, rig)
    assert out["us_per_deg"] == pytest.approx(US_PER_DEG, rel=0.1)
    assert out["delay_ms"] == pytest.approx(1000 * DEAD_S, abs=15)
    assert out["top_speed_deg_s"] == pytest.approx(TOP_DEG_S, rel=0.2)
    assert out["dead_band_us"] == 20
    assert out["verdict"] == "ok"
