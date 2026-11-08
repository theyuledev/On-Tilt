import math

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision.calib import project
from src.vision.plate_pose import (
    QuadTracker,
    find_plate_quad,
    plate_tilt,
    square,
    step_response,
    sweep_fit,
    tilt_report,
)
from tools.servo_test import plan

pytestmark = pytest.mark.unit
TICK = 0.01
SIZE = 305
# wide lenses like the overhead cam, plate off to the right of the frame
K = [[700.0, 0, 640], [0, 700.0, 400], [0, 0, 1]]
STANDARD = {"model": "standard", "K": K, "D": [-0.3, 0.1, 0, 0, -0.02]}
FISHEYE = {"model": "fisheye", "K": K, "D": [0.05, -0.02, 0.01, 0]}
T = np.array([80.0, -10.0, 380.0])


def rot(axis, deg):
    return cv2.Rodrigues(np.array(axis, float) * math.radians(deg))[0]


def pose(tip_y=0.0, tip_x=0.0):
    """Plate facing the cam as a diamond (corner 0 at the top), tipped about its own edges."""
    r = (
        np.diag([1.0, -1.0, -1.0])
        @ rot([0, 0, 1], -45)
        @ rot([0, 1, 0], tip_y)
        @ rot([1, 0, 0], tip_x)
    )
    return r, math.degrees(math.asin(r[2, 0])), math.degrees(math.asin(r[2, 1]))


def render(r, lens=STANDARD, balls=True):
    s = SIZE / 2
    edge = np.linspace(-s, s, 25)
    rim = np.concatenate(
        [
            np.c_[edge, np.full(25, s)],
            np.c_[np.full(25, s), -edge],
            np.c_[-edge, np.full(25, -s)],
            np.c_[np.full(25, -s), edge],
        ]
    )
    rvec, _ = cv2.Rodrigues(r)
    px = project(lens, np.c_[rim, np.zeros(len(rim))], rvec, T)
    img = np.full((800, 1280), 90, np.uint8)
    img += (np.random.default_rng(0).random(img.shape) * 8).astype(np.uint8)
    # fixed point coords so the corners land where the projection says, not on a pixel grid
    cv2.fillPoly(img, [(px * 16).round().astype(np.int32)], 225, cv2.LINE_AA, 4)
    if balls:
        for c, rad in (((60, 40, 0), 40), ((-20, -30, 0), 25), ((-90, 60, 0), 8)):
            b = project(lens, [c], rvec, T)[0]
            cv2.circle(img, tuple(b.round().astype(int)), rad, 30, -1)
    return img


@pytest.mark.parametrize("lens", [STANDARD, FISHEYE], ids=lambda c: c["model"])
def test_quad_and_tilt_come_back_from_a_rendered_plate(lens):
    r, tx, ty = pose(tip_y=6, tip_x=-4)
    quad = find_plate_quad(render(r, lens))
    assert quad is not None and quad.shape == (4, 2)
    assert np.argmin(quad[:, 1]) == 0 and quad[1, 0] > quad[3, 0]
    got_x, got_y, _, tvec = plate_tilt(quad, lens, SIZE)
    assert got_x == pytest.approx(tx, abs=0.5) and got_y == pytest.approx(ty, abs=0.5)
    assert np.allclose(tvec.ravel(), T, atol=8)


def test_positive_tilt_means_the_plus_edge_is_lower():
    for tip_y, tip_x in ((5, -3), (-5, 3)):
        r, tx, ty = pose(tip_y, tip_x)
        # camera z grows away from the cam, so the lower edge has the bigger z
        z = (square(SIZE) @ r.T + T)[:, 2]
        assert (tx > 0) == (z[[1, 2]].mean() > z[[0, 3]].mean())
        assert (ty > 0) == (z[[0, 1]].mean() > z[[2, 3]].mean())
        got_x, got_y, _, _ = plate_tilt(find_plate_quad(render(r)), STANDARD, SIZE)
        assert np.sign(got_x) == np.sign(tx) and np.sign(got_y) == np.sign(ty)
    assert pose(tip_y=5)[1] > 0


def test_small_tilts_keep_their_sign():
    for tip in (0.5, -0.5):
        r, tx, _ = pose(tip_y=tip)
        got, *_ = plate_tilt(find_plate_quad(render(r)), STANDARD, SIZE)
        assert got == pytest.approx(tx, abs=0.5)


def test_prev_keeps_the_corner_order_and_blank_frames_give_none():
    img = render(pose()[0])
    quad = find_plate_quad(img)
    rolled = np.roll(quad, 1, axis=0)
    assert np.allclose(find_plate_quad(img, prev=rolled), rolled, atol=0.5)
    assert find_plate_quad(np.full((800, 1280), 90, np.uint8)) is None


def test_tracker_rows_carry_tilt_and_corners():
    r, tx, ty = pose(tip_y=3)
    tracker = QuadTracker(STANDARD, SIZE)
    row = tracker.update(render(r), 1.0, 1500)
    assert len(row) == 11 and row[0] == 1.0
    assert row[1] == pytest.approx(tx, abs=0.5) and row[2] == pytest.approx(ty, abs=0.5)
    assert tracker.image is not None and tracker.quad.shape == (4, 2)


WN, ZETA = 15.0, 0.5


def h(t):
    """Unit step response of a second order lag, zero before t=0."""
    wd = WN * math.sqrt(1 - ZETA**2)
    env = np.exp(-ZETA * WN * t) * (np.cos(wd * t) + ZETA / math.sqrt(1 - ZETA**2) * np.sin(wd * t))
    return np.where(t > 0, 1 - env, 0.0)


def toy_plate(steps, rig, delay=0.03, fps=119, ack=0.002):
    tc = np.arange(len(steps)) * TICK
    p = np.array([us for us, _ in steps], float)
    commands = [(t, t + ack, us, 1500, ph) for t, (us, ph) in zip(tc, steps, strict=True)]
    tf = np.arange(0, tc[-1] + TICK, 1 / fps)
    y = np.zeros(len(tf))
    for k in np.flatnonzero(np.diff(p)) + 1:
        y += (p[k] - p[k - 1]) * h(tf - tc[k] - ack - delay)
    y = y / rig["us_per_deg_x"] + np.random.default_rng(2).normal(0, 0.02, len(tf))
    return commands, [(t, v, 0.0) for t, v in zip(tf, y, strict=True)]


def test_step_response_matches_the_analytic_curve():
    t = np.arange(-0.1, 1.0, 0.0005)
    y = 2.0 * h(t - 0.03)
    out = step_response(t, y, 0.0, 1.0, 0.001)
    fine = h(t[t > 0.03] - 0.03)
    t10, t90 = (t[t > 0.03][np.argmax(fine > q)] - 0.03 for q in (0.1, 0.9))
    assert out["delay"] == pytest.approx(0.03, abs=0.005)
    assert out["rise"] == pytest.approx(t90 - t10, rel=0.02)
    assert out["overshoot"] == pytest.approx(
        100 * math.exp(-math.pi * ZETA / math.sqrt(1 - ZETA**2)), abs=0.5
    )
    assert out["settle"] == pytest.approx(
        0.03 + (t[t > 0.03] - 0.03)[np.flatnonzero(abs(fine - 1) > 0.05)[-1] + 1], abs=0.002
    )
    assert out["change"] == pytest.approx(2.0, abs=0.01)
    assert out["speed"] == pytest.approx(
        np.polyfit(t[(y > 0.4) & (y < 1.6)], y[(y > 0.4) & (y < 1.6)], 1)[0], rel=0.05
    )
    assert step_response(t, np.zeros_like(t), 0.0, 1.0, 0.01) is None


def test_step_response_gives_up_on_a_sparse_step_nothing_crosses():
    # the last third sits exactly on the noise band so no sample is above it
    t = np.array([-0.05, 0.05, 0.1, 0.15])
    y = np.array([0.0, 0.06, 0.06, 0.06])
    assert step_response(t, y, 0.0, 1.0, 0.02) is None


def test_tilt_report_recovers_the_toy_plate():
    rig = load()["rig"]
    commands, tilts = toy_plate(plan(rig, 0, TICK), rig)
    out = tilt_report(commands, tilts, 0)
    t = np.arange(0, 2, 0.0001)
    t10, t90 = (t[np.argmax(h(t) > q)] for q in (0.1, 0.9))
    settle = t[np.flatnonzero(abs(h(t) - 1) > 0.05)[-1] + 1]
    # delay counts until the tilt clears 3 sigma, which this slow toy takes a while to do
    seen = t[np.argmax(h(t) > 3 * out["noise_deg"] / (100 / rig["us_per_deg_x"]))]
    assert out["delay_ms"] == pytest.approx(1000 * (0.03 + seen), abs=10)
    assert out["rise_ms"] == pytest.approx(1000 * (t90 - t10), rel=0.1)
    assert out["settle_ms"] == pytest.approx(1000 * settle, rel=0.1)
    assert out["overshoot_pct"] == pytest.approx(16.3, abs=2)
    assert out["deg_per_us"] == pytest.approx(1 / rig["us_per_deg_x"], rel=0.05)
    assert out["deg_per_us_sweep"] == pytest.approx(1 / rig["us_per_deg_x"], rel=0.1)
    assert out["fps"] == pytest.approx(119, rel=0.01) and out["dropped_frames"] == 0
    assert out["frames"] == len(tilts) and out["no_data"] == []


@pytest.mark.parametrize("kept", [{"find"}, {"find", "hold"}])
def test_tilt_report_on_a_cut_run_skips_the_phases_it_never_reached(kept):
    rig = load()["rig"]
    steps = [s for s in plan(rig, 0, TICK) if s[1] in kept]
    out = tilt_report(*toy_plate(steps, rig), 0)
    assert out["deg_per_us_sweep"] is None and out["delay_ms"] is None
    assert (out["noise_deg"] is None) == ("hold" not in kept)
    assert out["no_data"] == [p for p in ("hold", "sweep", "step") if p not in kept]


def test_sweep_fit_needs_two_samples():
    tc, p = np.array([0.0, 1.0, 2.0]), np.array([1500.0, 1600, 1700])
    phase = np.array(["hold", "sweep", "sweep"])
    assert sweep_fit(np.array([0.5, 1.5]), np.array([0.0, 1.0]), tc, p, phase) is None
    k, b = sweep_fit(np.array([0.5, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), tc, p, phase)
    assert k == pytest.approx(0.01) and b == pytest.approx(-15)


def test_tilt_report_counts_dropped_frames_and_the_y_axis():
    rig = load()["rig"]
    commands, tilts = toy_plate(plan(rig, 0, TICK), rig)
    del tilts[100:106]
    assert tilt_report(commands, tilts, 0)["dropped_frames"] == 1
    swapped = [(c[0], c[1], 1500, c[2], c[4]) for c in commands]
    out = tilt_report(swapped, [(t, 0.0, v) for t, v, _ in tilts], 1)
    assert out["deg_per_us"] == pytest.approx(1 / rig["us_per_deg_x"], rel=0.05)
