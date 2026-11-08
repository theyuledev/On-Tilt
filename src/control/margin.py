import numpy as np


def phase_margin_deg(
    kp: float, kd: float, delay_s: float, tau_s: float, g_mm_s2: float, ki: float = 0.0
) -> float:
    """Margin of the balance loop: PID (deg/mm, deg per mm/s, deg per mm s) on a rolling ball
    through a servo with first order lag tau_s and pure delay delay_s (dead time, camera, tick)."""
    w = np.logspace(-3, 4, 7001)
    k = 5 / 7 * g_mm_s2 * np.pi / 180  # mm/s^2 per degree of tilt
    mag = np.hypot(kp, kd * w - ki / w) * k / (w**2 * np.hypot(1, w * tau_s))
    # |L| only falls with w, so the crossover is a lookup on -log|L|
    wc = np.exp(np.interp(0.0, -np.log(mag), np.log(w)))
    lead = np.arctan2(kd * wc - ki / wc, kp)
    return float(np.degrees(lead - wc * delay_s - np.arctan(wc * tau_s)))
