import numpy as np

from src.vision.px_to_mm import SPAN

PASS_DEG = 45


def crossover_rad_s(kp: float, kd: float, tau_s: float, g_mm_s2: float, ki: float = 0.0) -> float:
    w = np.logspace(-3, 4, 7001)
    k = 5 / 7 * g_mm_s2 * np.pi / 180  # mm/s^2 per degree of tilt
    mag = np.hypot(kp, kd * w - ki / w) * k / (w**2 * np.hypot(1, w * tau_s))
    # |L| only falls with w, so the crossover is a lookup on -log|L|
    return float(np.exp(np.interp(0.0, -np.log(mag), np.log(w))))


def phase_margin_deg(
    kp: float, kd: float, delay_s: float, tau_s: float, g_mm_s2: float, ki: float = 0.0
) -> float:
    """Margin of the balance loop: PID (deg/mm, deg per mm/s, deg per mm s) on a rolling ball
    through a servo with first order lag tau_s and pure delay delay_s (dead time, camera, tick)."""
    wc = crossover_rad_s(kp, kd, tau_s, g_mm_s2, ki)
    lead = np.arctan2(kd * wc - ki / wc, kp)
    return float(np.degrees(lead - wc * delay_s - np.arctan(wc * tau_s)))


def balance_margin(cfg: dict) -> dict:
    c = cfg["control"]
    tick = 1 / c["sim_tick_hz"]
    # each tick takes the newest frame, so the track gets one entry per tick
    # unless the camera is slower than the loop
    sample = max(tick, 1 / cfg["rig"]["camera_fps"])
    vel_lag = SPAN / 2 * sample
    dead, ema = c["servo_dead_time_s"], c["vel_ema_s"]
    # the velocity lags only hit the d path, so charging them to the whole loop is on the safe side
    delay = dead + tick + vel_lag + ema
    kp, kd, ki = c["kp_deg_mm"], c["kd_deg_s_mm"], c["ki_deg_s_mm"]
    tau, g = c["servo_tau_s"], c["sim_g_mm_s2"]
    wc = crossover_rad_s(kp, kd, tau, g, ki)
    margin = phase_margin_deg(kp, kd, delay, tau, g, ki)
    return {
        "kp_deg_mm": kp,
        "kd_deg_s_mm": kd,
        "ki_deg_s_mm": ki,
        "dead_time_s": dead,
        "tick_s": tick,
        "vel_lag_s": vel_lag,
        "vel_ema_s": ema,
        "delay_s": delay,
        "crossover_rad_s": wc,
        "ring_period_s": 2 * np.pi / wc,
        "margin_deg": margin,
        "pass": margin >= PASS_DEG,
    }
