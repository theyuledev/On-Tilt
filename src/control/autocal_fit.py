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
    resid: float  # rms misfit of the doublet fits, mm


def wrap(deg: float) -> float:
    w = (deg + 180) % 360 - 180
    return 180.0 if w == -180 else w


def angle(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


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


def per_deg(g: float) -> float:
    return ROLL * g * math.sin(math.radians(1))


def double_integral(t, o) -> np.ndarray:
    i1 = np.concatenate([[0.0], np.cumsum((o[1:] + o[:-1]) / 2 * np.diff(t))])
    return np.concatenate([[0.0], np.cumsum((i1[1:] + i1[:-1]) / 2 * np.diff(t))])


def probe_fit(t, pos, tm, offs, t0: float):
    """pos = p0 + v0 tau + b tau^2/2 + E M, E the double integral of each servo offset column.
    Returns M (2 x columns of offs, mm/s^2 per us), b (mm/s^2) and the rms misfit (mm)."""
    tau, tmt = np.asarray(t) - t0, np.asarray(tm) - t0
    offs = np.asarray(offs, dtype=float)
    e = [np.interp(tau, tmt, double_integral(tmt, o)) for o in offs.T]
    a = np.column_stack([np.ones_like(tau), tau, tau * tau / 2, *e])
    coef, *_ = np.linalg.lstsq(a, pos, rcond=None)
    res = pos - a @ coef
    return coef[3:].T, coef[2], float(np.sqrt(np.mean(res**2)))


def doublet_fit(t, pos, tm, offs, t0: float, axis: int):
    m, b, rms = probe_fit(t, pos, tm, np.asarray(offs)[:, [axis]], t0)
    return m[:, 0], b, rms


def hold_accel(p, v, integ, c: dict) -> np.ndarray:
    w, z = c["hold_w_rad_s"], c["hold_zeta"]
    return -w * w * (p + v * c["hold_lead_s"]) - 2 * z * w * v - c["hold_ki"] * integ


def trim_step(m, b, axis: int) -> float:
    return float(-np.linalg.solve(m, b)[axis])


def stored_m(rig: dict, yaw_deg: float, g: float) -> np.ndarray:
    a = math.radians(-yaw_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return per_deg(g) * rot / np.array([rig["us_per_deg_x"], rig["us_per_deg_y"]])


def mapping(m, g: float, tol: float) -> tuple[float, bool, float, float]:
    a = angle(m[:, 0])
    mirror = mirror_verdict(a, angle(m[:, 1]), tol)
    kx, ky = (float(np.hypot(*m[:, j])) for j in (0, 1))
    return -a, mirror, per_deg(g) / kx, (-1 if mirror else 1) * per_deg(g) / ky


def agree(m, ref, deg: float, band) -> bool:
    for j in (0, 1):
        k = np.hypot(*m[:, j]) / np.hypot(*ref[:, j])
        if abs(wrap(angle(m[:, j]) - angle(ref[:, j]))) > deg or not band[0] <= k <= band[1]:
            return False
    return True
