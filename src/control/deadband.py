from dataclasses import dataclass

from src.contracts import ServoCommand
from src.control.pulse_map import center
from src.safety import limits


@dataclass(frozen=True)
class Axis:
    committed: float
    extreme: float  # furthest the target got the way it is heading
    direction: int = 0


def start(rig: dict) -> tuple[Axis, Axis]:
    cx, cy = center(rig)
    return Axis(cx, cx), Axis(cy, cy)


def _step(a: Axis, target: float, band: float, comp: float, flip: float, lo: float, hi: float):
    heading, extreme = a.direction, a.extreme
    against = (extreme - target) * heading if heading else abs(target - extreme)
    if against > flip:
        # turned round by more than the hysteresis, not just measurement jitter
        heading = -heading if heading else (1 if target > extreme else -1)
        extreme = target
    elif (target - extreme) * heading > 0:
        extreme = target
    # lead the servo by half the band the way it is heading, so a real turn round
    # jumps the pulse by a whole band and the servo really moves
    out = min(max(target + heading * comp / 2, lo), hi)
    # a pulse pinned at a limit always goes out, the lead cannot carry it over the band there,
    # and the servo is never left more than comp from the pulse safety approved
    hold = abs(out - a.committed) < band and lo < out < hi and abs(a.committed - target) <= comp
    return Axis(a.committed if hold else out, extreme, heading)


def invert(
    state: tuple[Axis, Axis], sent: ServoCommand | None, bypass: bool, cfg: dict
) -> tuple[tuple[Axis, Axis], ServoCommand | None]:
    """Runs on the pulses safety let through. Off, and forgets its place, while disarmed,
    killing or stale, so a ramp lands level and the next move starts from centre."""
    rig, c = cfg["rig"], cfg["control"]
    if sent is None or bypass:
        return start(rig), sent
    band, comp, flip = c["servo_dead_band_us"], c["db_comp_us"], c["db_flip_us"]
    x_lo, x_hi, y_lo, y_hi = limits(rig)
    x = _step(state[0], sent.x_us, band, comp, flip, x_lo, x_hi)
    y = _step(state[1], sent.y_us, band, comp, flip, y_lo, y_hi)
    return (x, y), ServoCommand(round(x.committed), round(y.committed))
