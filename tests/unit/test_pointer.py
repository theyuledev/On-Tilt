import math

import numpy as np
import pytest

from src.io.settings import load
from src.safety import limits
from src.vision.pointer import PairTracker, bright_blobs, pick_pair, report, track_pair
from tools.servo_test import plan

pytestmark = pytest.mark.unit
TICK = 0.02


@pytest.fixture
def rig():
    return load()["rig"]


def frame(angle_deg, radius=150, size=60, dot=8, extra=None, shift=(0, 0)):
    img = np.full((800, 1280), 40, np.uint8)
    img += (np.random.default_rng(0).random(img.shape) * 10).astype(np.uint8)
    th = math.radians(angle_deg)
    for sign in (1, -1):
        cx, cy = 640 + sign * radius * math.cos(th), 400 + sign * radius * math.sin(th)
        x, y = round(cx - size / 2), round(cy - size / 2)
        img[y : y + size, x : x + size] = 240
        dx, dy = round(cx) + shift[0], round(cy) + shift[1]
        img[dy - dot : dy + dot, dx - dot : dx + dot] = 30
    if extra:
        x, y, w, h = extra
        img[y : y + h, x : x + w] = 255
    return img


def test_bright_blobs_finds_the_two_notes_and_skips_small_dots():
    img = frame(30, extra=(1000, 700, 6, 6))
    blobs = bright_blobs(img)
    assert len(blobs) == 2 and math.dist(blobs[0], blobs[1]) == pytest.approx(300, abs=2)


def test_pick_pair_takes_the_blobs_that_moved():
    still = (20, 20, 300, 300)
    before = bright_blobs(frame(0, extra=still))
    after = bright_blobs(frame(4, extra=still))
    pair = pick_pair(before, after)
    assert pair is not None and math.dist(*pair) == pytest.approx(300, abs=2)
    assert all(p[0] > 400 for p in pair)
    assert pick_pair(before, before) is None


def test_track_pair_follows_the_notes_and_unwraps():
    pair = pick_pair(bright_blobs(frame(100)), bright_blobs(frame(104)))
    angle, _, _ = track_pair(frame(120), pair, prev=100)
    assert angle == pytest.approx(120, abs=0.5)
    pair = pick_pair(bright_blobs(frame(161)), bright_blobs(frame(165)))
    angle, _, _ = track_pair(frame(-10), pair, prev=165)
    assert angle == pytest.approx(170, abs=0.5)
    assert track_pair(frame(60), pair) is None


def test_plan_stays_in_startup_limits_and_has_every_phase(rig):
    steps = plan(rig, 0, TICK)
    lo, hi = limits(rig)[:2]
    assert all(lo <= us <= hi for us, _ in steps)
    phases = [p for _, p in steps]
    assert (
        phases[0] == "find"
        and "hold" in phases
        and "sweep" in phases
        and "step" in phases
        and "rev30" in phases
    )


def toy_servo(steps, rig, tau=0.04, delay=0.03, dead=6.0, noise=0.02, fps=119):
    """First-order lag + pure delay + dead band, sampled at camera frame times."""
    tc = np.arange(len(steps)) * TICK
    p = np.array([us for us, _ in steps], float)
    commands = [(t, t, us, 1500, ph) for t, (us, ph) in zip(tc, steps, strict=True)]
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
    assert abs(out["hysteresis_deg"]) < 0.5
    assert out["noise_band_us"] == pytest.approx(
        3 * out["noise_deg"] * rig["us_per_deg_x"], rel=0.1
    )
    assert out["delay_ms"] == pytest.approx(30, abs=10)
    # 20-80 % slope of a first order lag answering a 100 us step
    slope = 0.6 * (100 / rig["us_per_deg_x"]) / (0.04 * math.log(4))
    assert out["top_speed_deg_s"] == pytest.approx(slope, rel=0.3)
    assert out["dead_band_us"] == 8
    assert out["reversal_lag_ms"] == pytest.approx(30, abs=15)
    assert out["verdict"] == "ok"


def test_report_rejects_wide_dead_band(rig):
    out = report(*toy_servo(plan(rig, 0, TICK), rig, dead=40), rig)
    assert out["dead_band_us"] is None
    assert out["verdict"] == "reject"


def test_track_pair_gives_up_when_a_note_jumps_more_than_the_spacing():
    pair = pick_pair(bright_blobs(frame(0)), bright_blobs(frame(4)))
    assert track_pair(frame(4, radius=400), pair) is None
    assert track_pair(frame(30), pair) is not None


def test_pair_tracker_picks_near_the_top_of_the_nudge_then_follows():
    tracker = PairTracker(top=1650)
    assert tracker.update(frame(0), 0.0, 1500) is None
    assert tracker.update(frame(4), 0.1, 1630) is None and tracker.pair is None
    assert tracker.update(frame(4), 0.2, 1642) is None and tracker.pair is not None
    t, angle = tracker.update(frame(6), 0.3, 1650)
    assert t == 0.3 and angle == pytest.approx(6, abs=0.5)


def test_track_pair_centres_on_the_dark_dot_or_falls_back_to_the_note():
    pair = pick_pair(bright_blobs(frame(0)), bright_blobs(frame(4)))
    _, *pts = track_pair(frame(0, shift=(6, -3)), pair)
    left, right = sorted(pts)
    assert left == pytest.approx((490 + 6 - 0.5, 400 - 3 - 0.5), abs=0.3)
    assert right == pytest.approx((790 + 6 - 0.5, 400 - 3 - 0.5), abs=0.3)
    _, *pts = track_pair(frame(0, dot=0), pair)
    assert sorted(pts)[0] == pytest.approx((490 - 0.5, 400 - 0.5), abs=0.3)
