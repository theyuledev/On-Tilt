import math

import numpy as np
import pytest

from src.io.settings import load
from src.safety import limits
from src.vision.pointer import horn_angle, motion_roi, report
from tools.servo_test import plan

pytestmark = pytest.mark.unit
TICK = 0.02


@pytest.fixture
def rig():
    return load()["rig"]


def frame(angle_deg, radius=150, size=60, dot=8, extra=None):
    img = np.full((800, 1280), 40, np.uint8)
    img += (np.random.default_rng(0).random(img.shape) * 10).astype(np.uint8)
    th = math.radians(angle_deg)
    for sign in (1, -1):
        cx, cy = 640 + sign * radius * math.cos(th), 400 + sign * radius * math.sin(th)
        x, y = round(cx - size / 2), round(cy - size / 2)
        img[y : y + size, x : x + size] = 240
        img[round(cy) - dot : round(cy) + dot, round(cx) - dot : round(cx) + dot] = 30
    if extra:
        x, y, w, h = extra
        img[y : y + h, x : x + w] = 255
    return img


def test_horn_angle_finds_two_squares():
    angle, p1, p2 = horn_angle(frame(30))
    assert abs(angle - 30) < 0.5 or abs(angle + 150) < 0.5
    assert math.dist(p1, p2) == pytest.approx(300, abs=2)


def test_horn_angle_unwraps_against_previous():
    angle, _, _ = horn_angle(frame(120), prev=100)
    assert angle == pytest.approx(120, abs=0.5)
    angle, _, _ = horn_angle(frame(-10), prev=175)
    assert angle == pytest.approx(170, abs=0.5)


def test_roi_rejects_bright_distractor_outside_it():
    img = frame(0, extra=(20, 20, 300, 300))
    angle, _, _ = horn_angle(img, roi=(300, 200, 700, 400), prev=0)
    assert angle == pytest.approx(0, abs=0.5)
    assert motion_roi(frame(0), frame(0)) is None
    x, y, w, h = motion_roi(frame(0), frame(40))
    assert x < 640 - 150 < 640 + 150 < x + w and y < 400 < y + h


def test_plan_stays_in_startup_limits_and_has_every_phase(rig):
    steps = plan(rig, 0, TICK)
    lo, hi = limits(rig)[:2]
    assert all(lo <= us <= hi for us, _ in steps)
    phases = [p for _, p in steps]
    assert phases[0] == "hold" and "sweep" in phases and "step" in phases and "rev30" in phases


def toy_servo(steps, rig, tau=0.04, delay=0.03, dead=6.0, noise=0.02, fps=119):
    """First-order lag + pure delay + dead band, sampled at camera frame times."""
    tc = np.arange(len(steps)) * TICK
    p = np.array([us for us, _ in steps], float)
    commands = [(t, us, 1500, ph) for t, (us, ph) in zip(tc, steps, strict=True)]
    rng = np.random.default_rng(1)
    target = pos = p[0]
    angles = []
    for t in np.arange(0, tc[-1] + TICK, 1 / fps):
        cmd = p[max(np.searchsorted(tc, t - delay, side="right") - 1, 0)]
        if abs(cmd - target) > dead:
            target = cmd
        pos += (target - pos) * (1 - math.exp(-1 / (fps * tau)))
        angles.append((t, (pos - 1500) / rig["us_per_deg_x"] + rng.normal(0, noise)))
    return commands, angles


def test_report_recovers_toy_servo(rig):
    out = report(*toy_servo(plan(rig, 0, TICK), rig), rig)
    assert out["us_per_deg"] == pytest.approx(rig["us_per_deg_x"], rel=0.1)
    assert abs(out["center_offset_deg"]) < 0.5
    assert out["delay_ms"] == pytest.approx(30, abs=10)
    assert 40 < out["top_speed_deg_s"] < 100
    assert out["dead_band_us"] == 8
    assert out["reversal_lag_ms"] == pytest.approx(30, abs=15)
    assert out["verdict"] == "ok"


def test_report_rejects_wide_dead_band(rig):
    out = report(*toy_servo(plan(rig, 0, TICK), rig, dead=40), rig)
    assert out["dead_band_us"] is None
    assert out["verdict"] == "reject"
