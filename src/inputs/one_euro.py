import math


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1 / (2 * math.pi * cutoff_hz)
    return 1 / (1 + tau / dt)


def one_euro(state, x: float, t: float, cfg: dict) -> tuple[float, tuple]:
    """One euro filter (Casiez 2012): heavy smoothing when still, little when moving fast."""
    if state is None:
        return x, (x, 0.0, t)
    x_prev, dx_prev, t_prev = state
    dt = t - t_prev
    if dt <= 0:
        return x_prev, state
    dx = dx_prev + _alpha(cfg["hand_d_cutoff_hz"], dt) * ((x - x_prev) / dt - dx_prev)
    fc = cfg["hand_min_cutoff_hz"] + cfg["hand_beta"] * abs(dx)
    x_hat = x_prev + _alpha(fc, dt) * (x - x_prev)
    return x_hat, (x_hat, dx, t)
