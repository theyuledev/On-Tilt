from dataclasses import dataclass

from src.contracts import ServoCommand
from src.control.pulse_map import center

LEVEL_HOLD_S = 0.04  # two 20 ms servo pwm periods, so the level pulse really goes out


@dataclass
class SafetyState:
    armed: bool = False
    killing: bool = False
    x_us: float | None = None
    y_us: float | None = None
    level_s: float = 0.0

    def arm(self) -> None:
        # a kill runs to the end, no re-arming half way down the ramp
        if not self.killing:
            self.armed = True

    def kill(self) -> None:
        if self.armed:
            self.killing = True


def _toward(cur: float, target: float, max_step: float) -> float:
    if abs(target - cur) <= max_step:
        return target
    return cur + max_step if target > cur else cur - max_step


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def limits(rig: dict) -> tuple[float, float, float, float]:
    cx, cy = center(rig)
    lim_x, lim_y = rig["limit_x_us"], rig["limit_y_us"]
    if rig["startup_flag"]:
        lim_x = min(lim_x, rig["startup_range_us"])
        lim_y = min(lim_y, rig["startup_range_us"])
    return cx - lim_x, cx + lim_x, cy - lim_y, cy + lim_y


def step(
    state: SafetyState, target: ServoCommand, stale: bool, dt: float, rig: dict
) -> ServoCommand | None:
    if not state.armed:
        return None

    cx, cy = center(rig)
    if state.x_us is None:
        state.x_us, state.y_us = cx, cy

    if state.killing or stale:
        tx, ty, rate = cx, cy, rig["kill_ramp_us_s"]
    else:
        x_lo, x_hi, y_lo, y_hi = limits(rig)
        tx = _clamp(target.x_us, x_lo, x_hi)
        ty = _clamp(target.y_us, y_lo, y_hi)
        rate = rig["slew_us_s"]

    state.x_us = _toward(state.x_us, tx, rate * dt)
    state.y_us = _toward(state.y_us, ty, rate * dt)

    out = ServoCommand(x_us=round(state.x_us), y_us=round(state.y_us))

    # last pulses out are the level one, servos hold there once pulses stop
    if state.killing and (state.x_us, state.y_us) == (cx, cy):
        state.level_s += dt
        if state.level_s >= LEVEL_HOLD_S:
            state.armed = state.killing = False
            state.x_us = state.y_us = None
            state.level_s = 0.0

    return out
