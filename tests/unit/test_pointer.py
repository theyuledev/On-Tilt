import math

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.safety import limits
from src.vision.pointer import PairTracker, moved_pair, report
from tools.servo_test import plan

pytestmark = pytest.mark.unit
TICK = 0.02


@pytest.fixture
def rig():
    return load()["rig"]


def background():
    return 40 + (np.random.default_rng(0).random((800, 1280)) * 10).astype(np.uint8)


def square(mx, my, hc, tilt):
    """Maps a local (u, v) in [-1, 1] to the pixel of a turned square centred at (mx, my)."""
    c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
    return lambda u, v: (round(mx + hc * (u * c - v * s)), round(my + hc * (u * s + v * c)))


def frame(angle_deg, radius=100, size=60, dot=8, extra=(), shift=(0, 0), tilt=0, split=False):
    """Two notes on a horn: squares turned by tilt, with a dark dot; split cuts each in two."""
    img = background()
    th = math.radians(angle_deg)
    for sign in (1, -1):
        cx, cy = 640 + sign * radius * math.cos(th), 400 + sign * radius * math.sin(th)
        hc = (size - 1) / 2
        at = square(round(cx - size / 2) + hc, round(cy - size / 2) + hc, hc, tilt)
        corners = np.array([at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)], np.int32)
        cv2.fillConvexPoly(img, corners, 240)
        if split:
            cv2.line(img, at(-1, -0.5), at(1, -0.5), 30, 3)
        dx, dy = round(cx) + shift[0], round(cy) + shift[1]
        img[dy - dot : dy + dot, dx - dot : dx + dot] = 30
    for x, y, w, h in extra:
        img[y : y + h, x : x + w] = 255
    return img


def seeded(a0=0, a1=4, **kw):
    tracker = PairTracker(top=1650)
    assert tracker.update(frame(a0, **kw), 0.0, 1500) is None
    assert tracker.update(frame(a1, **kw), 0.1, 1650) is None and tracker.pair is not None
    return tracker


def angle(tracker, img, t=1.0):
    row = tracker.update(img, t, 1500)
    return None if row is None else row[1]


def test_moved_pair_takes_the_moved_notes_not_the_shifted_board_or_the_led():
    board, led = (20, 20, 300, 300), (1000, 700, 6, 6)
    pair = moved_pair(frame(0, extra=[board]), frame(4, extra=[(23, 20, 300, 300), led]))
    assert pair is not None and math.dist(*pair) == pytest.approx(200, abs=15)
    assert all(400 < p[0] < 800 for p in pair)
    assert moved_pair(frame(0, extra=[board]), frame(0, extra=[board, led])) is None


def test_moved_pair_skips_a_moving_blob_far_from_the_notes():
    pair = moved_pair(frame(0, extra=[(1000, 700, 40, 40)]), frame(4, extra=[(1010, 700, 40, 40)]))
    assert pair is not None and all(p[0] < 800 for p in pair)


def test_moved_pair_skips_two_blobs_too_close_to_be_a_horn():
    assert (
        moved_pair(frame(0, radius=15, size=24, dot=0), frame(20, radius=15, size=24, dot=0))
        is None
    )


def test_pair_tracker_picks_near_the_top_of_the_nudge_then_follows():
    tracker = PairTracker(top=1650)
    assert tracker.update(frame(0), 0.0, 1500) is None
    assert tracker.update(frame(4), 0.1, 1630) is None and tracker.pair is None
    assert tracker.update(frame(4), 0.2, 1642) is None and tracker.pair is not None
    assert tracker.seed is not None and tracker.image is not None
    t, a = tracker.update(frame(6), 0.3, 1650)
    assert t == 0.3 and a == pytest.approx(6, abs=0.5)


def test_tracker_follows_the_notes_and_unwraps():
    assert angle(seeded(100, 104), frame(120)) == pytest.approx(120, abs=0.5)
    assert angle(seeded(161, 165), frame(-10)) == pytest.approx(170, abs=0.5)


@pytest.mark.parametrize(
    "kw", [{"tilt": 45}, {"split": True, "dot": 0}, {"tilt": 30, "split": True}]
)
def test_tracker_handles_turned_and_bent_notes(kw):
    tracker = seeded(**kw)
    for a in (10, 30, 45, 60):
        assert angle(tracker, frame(a, **kw)) == pytest.approx(a, abs=0.7)


def test_tracker_keeps_the_lock_through_a_noisy_frame_without_notes():
    tracker = seeded()
    pair = tracker.pair
    assert angle(tracker, background()) is None and tracker.pair == pair
    assert angle(tracker, frame(8)) == pytest.approx(8, abs=0.5)


def test_tracker_centres_on_the_dark_dot_or_falls_back_to_the_note():
    tracker = seeded()
    tracker.update(frame(0, shift=(6, -3)), 1.0, 1500)
    left, right = sorted(tracker.pair)
    assert left == pytest.approx((540 + 6 - 0.5, 400 - 3 - 0.5), abs=0.3)
    assert right == pytest.approx((740 + 6 - 0.5, 400 - 3 - 0.5), abs=0.3)
    tracker.update(frame(0, dot=0), 1.1, 1500)
    assert sorted(tracker.pair)[0] == pytest.approx((540 - 0.5, 400 - 0.5), abs=0.3)


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
    assert out["verdict"] == "ok" and out["no_data"] == []


@pytest.mark.parametrize("kept", [{"find"}, {"find", "hold"}])
def test_report_on_a_cut_run_skips_the_phases_it_never_reached(rig, kept):
    steps = [s for s in plan(rig, 0, TICK) if s[1] in kept]
    out = report(*toy_servo(steps, rig), rig)
    for k in ("us_per_deg", "hysteresis_deg", "noise_band_us", "delay_ms", "dead_band_us"):
        assert out[k] is None
    assert (out["noise_deg"] is None) == ("hold" not in kept)
    assert out["no_data"] == [p for p in ("hold", "sweep", "step") if p not in kept]


def test_report_rejects_wide_dead_band(rig):
    out = report(*toy_servo(plan(rig, 0, TICK), rig, dead=40), rig)
    assert out["dead_band_us"] is None
    assert out["verdict"] == "reject"
