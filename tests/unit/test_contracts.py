import json

import pytest

from src.contracts import (
    BallState,
    Intent,
    PlateCommand,
    PlateState,
    RunLogRecord,
    ServoCommand,
)

pytestmark = pytest.mark.unit


def make_record():
    return RunLogRecord(
        t_s=12.5,
        mode="hand",
        armed=True,
        ball=BallState(x_mm=10.0, y_mm=-4.5, vx_mm_s=120.0, vy_mm_s=-3.0, valid=True, t_s=12.49),
        intent=Intent(x=0.25, y=-1.0, valid=True, t_s=12.48),
        plate_cmd=PlateCommand(tilt_x_deg=3.0, tilt_y_deg=-12.0),
        servo_cmd=ServoCommand(x_us=1580, y_us=1150),
        plate_state=PlateState(sent_x_deg=3.0, sent_y_deg=-12.0, est_x_deg=2.1, est_y_deg=-9.4),
    )


def test_run_log_round_trip():
    rec = make_record()
    line = json.dumps(rec.to_dict())
    assert RunLogRecord.from_dict(json.loads(line)) == rec


def test_messages_are_immutable():
    rec = make_record()
    with pytest.raises(AttributeError):
        rec.ball.x_mm = 0.0
