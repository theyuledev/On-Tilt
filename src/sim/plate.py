import math
from collections import deque

from src.contracts import PlateCommand, PlateState, ServoCommand
from src.control.pulse_map import center, pulse_to_tilt


def _follow(pos: float, target: float, dt: float, control: dict) -> float:
    # inside half the dead band the servo holds wherever it stopped
    if abs(target - pos) <= control["servo_dead_band_us"] / 2:
        return pos
    step = (target - pos) * (1 - math.exp(-dt / control["servo_tau_s"]))
    cap = control["servo_top_speed_us_s"] * dt
    return pos + min(max(step, -cap), cap)


class SimActuator:
    """Hobby servo per axis: dead time, first order lag capped at top speed, dead band."""

    def __init__(self, rig: dict, control: dict):
        self.rig = rig
        self.control = control
        self.pos = list(center(rig))
        self.target = list(self.pos)
        self.pending: deque[tuple[float, ServoCommand]] = deque()
        self.sent = PlateCommand(0.0, 0.0)
        self.t: float | None = None

    def send(self, cmd: ServoCommand | None, now: float) -> PlateState:
        if cmd is not None:
            self.pending.append((now, cmd))
            self.sent = pulse_to_tilt(cmd, self.rig)
        while self.pending and now - self.pending[0][0] >= self.control["servo_dead_time_s"]:
            cmd = self.pending.popleft()[1]
            self.target = [cmd.x_us, cmd.y_us]

        dt = 0.0 if self.t is None else now - self.t
        self.t = now
        self.pos = [
            _follow(p, t, dt, self.control) for p, t in zip(self.pos, self.target, strict=True)
        ]
        est = pulse_to_tilt(ServoCommand(round(self.pos[0]), round(self.pos[1])), self.rig)
        return PlateState(
            sent_x_deg=self.sent.tilt_x_deg,
            sent_y_deg=self.sent.tilt_y_deg,
            est_x_deg=est.tilt_x_deg,
            est_y_deg=est.tilt_y_deg,
        )
