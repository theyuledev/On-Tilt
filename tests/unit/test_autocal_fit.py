import math

import numpy as np
import pytest

from src.control.autocal_fit import (
    agree,
    double_integral,
    doublet_fit,
    hold_accel,
    mapping,
    mirror_verdict,
    per_deg,
    probe_fit,
    second_amp,
    stored_m,
    trim_step,
)
from src.vision.px_to_mm import ball_state
from src.vision.tracker import BallPx

M = np.array([[0.9, -0.4], [0.3, 1.1]])
B = np.array([12.0, -7.0])


def synthetic(m, b, offs):
    t = np.linspace(0, 1.2, 241)
    e = np.column_stack([double_integral(t, o) for o in offs.T])
    pos = np.array([3.0, -2.0]) + np.outer(t, [20.0, -5.0]) + np.outer(t * t / 2, b) + e @ m.T
    return t, pos


def test_probe_fit_recovers_the_map_and_the_drift():
    t = np.linspace(0, 1.2, 241)
    offs = np.c_[100 * np.sign(np.sin(8 * t)), 100 * np.sign(np.cos(5 * t))]
    _, pos = synthetic(M, B, offs)
    m, b, rms = probe_fit(t + 4.0, pos, t + 4.0, offs, 4.0)
    assert m == pytest.approx(M)
    assert b == pytest.approx(B)
    assert rms == pytest.approx(0, abs=1e-6)


def test_doublet_fit_reads_one_column():
    t = np.linspace(0, 1.2, 241)
    offs = np.c_[np.zeros_like(t), np.where(t < 0.6, 60.0, -60.0)]
    _, pos = synthetic(M, B, offs)
    col, b, _ = doublet_fit(t, pos, t, offs, 0.0, 1)
    assert col == pytest.approx(M[:, 1])
    assert b == pytest.approx(B)


def test_trim_step_undoes_a_level_error():
    du = np.array([15.0, -25.0])
    assert trim_step(M, -M @ du, 0) == pytest.approx(15)
    assert trim_step(M, -M @ du, 1) == pytest.approx(-25)


def test_hold_pulls_the_ball_back():
    c = dict(hold_w_rad_s=2.0, hold_zeta=0.9, hold_lead_s=0.1, hold_ki=4.0)
    a = hold_accel(np.array([10.0, 0.0]), np.array([5.0, 0.0]), np.array([1.0, 0.0]), c)
    assert a == pytest.approx([-4 * (10 + 0.5) - 3.6 * 5 - 4, 0])


@pytest.mark.parametrize("mirror", [False, True])
def test_mapping_round_trip_through_px_to_mm(mirror):
    rig = {"us_per_deg_x": 57.0, "us_per_deg_y": -57.0 if mirror else 57.0}
    m = stored_m(rig, 37.0, 9810)
    yaw, mir, ux, uy = mapping(m, 9810, 20)
    assert (yaw, mir, ux, uy) == pytest.approx(
        (37, mirror, rig["us_per_deg_x"], rig["us_per_deg_y"])
    )
    assert np.hypot(*m[:, 0]) == pytest.approx(per_deg(9810) / 57)
    # a +x pulse pushes the ball along the found plate +x
    ux, uy = m[:, 0] / np.hypot(*m[:, 0])
    cfg = {"plate_centre_px": (700, 380), "px_per_mm": 2.6, "plate_yaw_deg": yaw}
    cfg["velocity_span_frames"] = 4
    px = BallPx(700 + 2.6 * 50 * ux, 380 - 2.6 * 50 * uy, 10, True, 0.0)
    b = ball_state(px, [], cfg)
    assert (b.x_mm, b.y_mm) == pytest.approx((50, 0), abs=1e-6)


def test_agree_checks_angle_and_gain_per_column():
    ref = stored_m({"us_per_deg_x": 100.0, "us_per_deg_y": 100.0}, 0.0, 9810)
    assert agree(ref * 1.2, ref, 20, (0.5, 1.5))
    assert not agree(ref * 2.0, ref, 20, (0.5, 1.5))
    assert not agree(-ref, ref, 20, (0.5, 1.5))
    a = math.radians(30)
    turn = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    assert not agree(turn @ ref, ref, 20, (0.5, 1.5))


def test_axes_not_square_stop_the_fit():
    with pytest.raises(ValueError):
        mirror_verdict(10, 20, 20)


def test_second_amp_keeps_the_travel_under_the_cap():
    assert second_amp(30, 250, 0.6, 60, 100) == pytest.approx(30 * 60 / 90)
    assert second_amp(30, 50, 0.6, 60, 100) == 100
