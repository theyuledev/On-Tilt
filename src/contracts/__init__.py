"""Messages passed between the parts of the app.

Plate frame: origin at the centre, +x right and +y up in the overhead image.
A positive tilt lowers that axis's + edge, so the ball rolls toward +.
Times are seconds on time.monotonic(). Intent x, y run from -1 to 1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BallState:
    x_mm: float
    y_mm: float
    vx_mm_s: float
    vy_mm_s: float
    valid: bool
    t_s: float


@dataclass(frozen=True, slots=True)
class Intent:
    x: float
    y: float
    valid: bool
    t_s: float


@dataclass(frozen=True, slots=True)
class PlateCommand:
    tilt_x_deg: float
    tilt_y_deg: float


@dataclass(frozen=True, slots=True)
class ServoCommand:
    x_us: int
    y_us: int


@dataclass(frozen=True, slots=True)
class PlateState:
    sent_x_deg: float
    sent_y_deg: float
    est_x_deg: float
    est_y_deg: float


@dataclass(frozen=True, slots=True)
class RunLogRecord:
    t_s: float
    mode: str
    armed: bool
    ball: BallState
    intent: Intent
    plate_cmd: PlateCommand
    servo_cmd: ServoCommand
    plate_state: PlateState

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunLogRecord:
        return cls(
            t_s=d["t_s"],
            mode=d["mode"],
            armed=d["armed"],
            ball=BallState(**d["ball"]),
            intent=Intent(**d["intent"]),
            plate_cmd=PlateCommand(**d["plate_cmd"]),
            servo_cmd=ServoCommand(**d["servo_cmd"]),
            plate_state=PlateState(**d["plate_state"]),
        )
