from dataclasses import dataclass

from src.contracts import ServoCommand
from src.control.pulse_map import center


@dataclass
class SafetyState:
    armed: bool = False
    killing: bool = False
    x_us: float | None = None
    y_us: float | None = None

    def arm(self) -> None:
        self.armed = True
        self.killing = False

    def kill(self) -> None:
        if self.armed:
            self.killing = True


def _toward(cur: float, target: float, max_step: float) -> float:
    if abs(target - cur) <= max_step:
        return target
    return cur + max_step if target > cur else cur - max_step


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def step(
    state: SafetyState, target: ServoCommand, stale: bool, dt: float, rig: dict
) -> ServoCommand | None:
    if not state.armed:
        return None

    cx, cy = center(rig)
    if state.x_us is None:
        state.x_us, state.y_us = cx, cy

    lim_x, lim_y = rig["limit_x_us"], rig["limit_y_us"]
    if rig["startup_flag"]:
        lim_x = min(lim_x, rig["startup_range_us"])
        lim_y = min(lim_y, rig["startup_range_us"])

    if state.killing or stale:
        tx, ty, rate = cx, cy, rig["kill_ramp_us_s"]
    else:
        tx = _clamp(target.x_us, cx - lim_x, cx + lim_x)
        ty = _clamp(target.y_us, cy - lim_y, cy + lim_y)
        rate = rig["slew_us_s"]

    state.x_us = _toward(state.x_us, tx, rate * dt)
    state.y_us = _toward(state.y_us, ty, rate * dt)

    out = ServoCommand(x_us=round(state.x_us), y_us=round(state.y_us))

    # last pulse out is the level one, servos hold there once pulses stop
    if state.killing and (state.x_us, state.y_us) == (cx, cy):
        state.armed = state.killing = False
        state.x_us = state.y_us = None

    return out
