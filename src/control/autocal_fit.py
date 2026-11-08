import math
from dataclasses import dataclass

import numpy as np

from src.control import ROLL


@dataclass(frozen=True, slots=True)
class Calibration:
    trim_x_us: float
    trim_y_us: float
    us_per_deg_x: float
    us_per_deg_y: float  # negative when the image is mirrored
    plate_yaw_deg: float
    mirror: bool
    k_one_x: float  # one amplitude estimate, mm/s^2 per us
    k_one_y: float
    friction_x: float  # slope intercept, mm/s^2
    friction_y: float
    resid: float  # rms misfit of the 8 accels to the linear model, mm/s^2


def wrap(deg: float) -> float:
    w = (deg + 180) % 360 - 180
    return 180.0 if w == -180 else w


def angle(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.hypot(*v)


def mirror_verdict(a_deg: float, b_deg: float, tol: float) -> bool:
    d = wrap(b_deg - a_deg)
    if abs(d - 90) <= tol:
        return False
    if abs(d + 90) <= tol:
        return True
    raise ValueError(f"axes are {d:.0f} deg apart, expected +-90")


def second_amp(a1_us: float, accel1: float, pulse_s: float, travel_mm: float, max_us: float):
    # pulse + brake leaves the ball a T^2 away, keep that under the travel cap
    travel = abs(accel1) * pulse_s**2
    return min(max_us, a1_us * travel_mm / travel) if travel > 0 else max_us


def slope(amps, mags) -> tuple[float, float]:
    k, c = np.polyfit(np.asarray(amps, dtype=float), np.asarray(mags, dtype=float), 1)
    return float(k), float(c)


def trim_offsets(a_res, kx: float, ex, ky: float, ey) -> np.ndarray:
    # a_res = -kx ux ex - ky uy ey
    m = -np.c_[kx * np.asarray(ex), ky * np.asarray(ey)]
    return np.linalg.solve(m, np.asarray(a_res, dtype=float))


def solve(acc: dict, amps: tuple[float, float], old_trims, band_us: float, g: float, tol: float):
    """acc[(axis, sign, i)] is the flat frame accel (mm/s^2, y up) of the pulse at amps[i]."""
    e = {ax: unit(acc[(ax, 1, 1)] - acc[(ax, -1, 1)]) for ax in "xy"}
    a, b = angle(e["x"]), angle(e["y"])
    mirror = mirror_verdict(a, b, tol)
    k, k_one, fric = {}, {}, {}
    for ax in "xy":
        mags = [float(np.dot(acc[(ax, 1, i)] - acc[(ax, -1, i)], e[ax]) / 2) for i in (0, 1)]
        k[ax], fric[ax] = slope(amps, mags)
        k_one[ax] = mags[1] / (amps[1] - band_us / 2)
    a_res = np.mean([acc[(ax, s, 1)] for ax in "xy" for s in (1, -1)], axis=0)
    u = trim_offsets(a_res, k["x"], e["x"], k["y"], e["y"])
    miss = [
        acc[(ax, s, i)] - a_res - s * (k[ax] * amps[i] + fric[ax]) * e[ax]
        for ax in "xy"
        for s in (1, -1)
        for i in (0, 1)
    ]
    per_deg = ROLL * g * math.sin(math.radians(1))
    return Calibration(
        trim_x_us=old_trims[0] + float(u[0]),
        trim_y_us=old_trims[1] + float(u[1]),
        us_per_deg_x=per_deg / k["x"],
        us_per_deg_y=-per_deg / k["y"] if mirror else per_deg / k["y"],
        plate_yaw_deg=-a,
        mirror=mirror,
        k_one_x=k_one["x"],
        k_one_y=k_one["y"],
        friction_x=fric["x"],
        friction_y=fric["y"],
        resid=float(np.sqrt(np.mean(np.sum(np.square(miss), axis=1)))),
    )
