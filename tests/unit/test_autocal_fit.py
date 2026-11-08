import math

import numpy as np
import pytest

from src.control.autocal_fit import mirror_verdict, second_amp, solve, trim_offsets
from src.sim.fit import accel_vec
from src.vision.px_to_mm import ball_state
from src.vision.tracker import BallPx


def flat(yaw_deg: float, plate_vec, mirror: bool = False) -> np.ndarray:
    x, y = plate_vec
    y = -y if mirror else y
    a = math.radians(-yaw_deg)
    return np.array([x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)])


def test_accel_vec_recovers_a_quadratic_inside_the_window():
    t = np.linspace(0, 2, 201)
    x = 3 + 20 * (t - 0.5) + 0.5 * 40 * (t - 0.5) ** 2
    y = -1 - 5 * (t - 0.5) + 0.5 * -25 * (t - 0.5) ** 2
    junk = (t < 0.5) | (t > 1.2)
    x[junk], y[junk] = 999.0, -999.0
    assert accel_vec(t, x, y, 0.5, 1.2) == pytest.approx([40, -25])


def test_trim_solve_round_trip():
    ex, ey = flat(37, (1, 0)), flat(37, (0, 1), mirror=True)
    u = np.array([15.0, -25.0])
    a_res = -2.1 * u[0] * ex - 4.3 * u[1] * ey
    assert trim_offsets(a_res, 2.1, ex, 4.3, ey) == pytest.approx(u)


def accels(yaw: float, mirror: bool, k: float, amps, trim_u, fric: float = 0.0) -> dict:
    e = {"x": flat(yaw, (1, 0)), "y": flat(yaw, (0, 1), mirror)}
    a_res = -k * trim_u[0] * e["x"] - k * trim_u[1] * e["y"]
    return {
        (ax, s, i): a_res + s * (k * amps[i] - fric) * e[ax]
        for ax in "xy"
        for s in (1, -1)
        for i in (0, 1)
    }


@pytest.mark.parametrize("mirror", [False, True])
def test_yaw_and_mirror_round_trip_through_px_to_mm(mirror):
    k = 122.3 / 57
    cal = solve(accels(37, mirror, k, (30, 80), (6, -9), fric=15), (30, 80), (0, 0), 20, 9810, 20)
    assert cal.mirror is mirror
    assert cal.plate_yaw_deg == pytest.approx(37)
    assert cal.us_per_deg_x == pytest.approx(57, rel=1e-3)
    assert cal.us_per_deg_y == pytest.approx(-57 if mirror else 57, rel=1e-3)
    assert (cal.trim_x_us, cal.trim_y_us) == pytest.approx((6, -9))
    assert cal.friction_x == pytest.approx(-15)
    # a ball off along the found +x axis reads as plate +x
    ux, uy = flat(37, (1, 0))
    cfg = {"plate_centre_px": (700, 380), "px_per_mm": 2.6, "plate_yaw_deg": cal.plate_yaw_deg}
    cfg["velocity_span_frames"] = 4
    px = BallPx(700 + 2.6 * 50 * ux, 380 - 2.6 * 50 * uy, 10, True, 0.0)
    b = ball_state(px, [], cfg)
    assert (b.x_mm, b.y_mm) == pytest.approx((50, 0), abs=1e-6)


def test_axes_not_square_stop_the_fit():
    with pytest.raises(ValueError):
        mirror_verdict(10, 20, 20)


def test_second_amp_keeps_the_travel_under_the_cap():
    assert second_amp(30, 250, 0.6, 60, 100) == pytest.approx(30 * 60 / 90)
    assert second_amp(30, 50, 0.6, 60, 100) == 100
