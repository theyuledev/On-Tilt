import pytest

from src.contracts import (
    BallState,
    Intent,
    PlateCommand,
    PlateState,
    RunLogRecord,
    ServoCommand,
)
from src.games import ghost

pytestmark = pytest.mark.unit


def record(t, x, y, valid=True):
    return RunLogRecord(
        t_s=t,
        mode="hand",
        armed=True,
        ball=BallState(x_mm=x, y_mm=y, vx_mm_s=0.0, vy_mm_s=0.0, valid=valid, t_s=t),
        intent=Intent(x=0.0, y=0.0, valid=True, t_s=t),
        plate_cmd=PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0),
        servo_cmd=ServoCommand(x_us=1500, y_us=1500),
        plate_state=PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0),
    )


def test_save_load_round_trip(tmp_path):
    recs = [record(10.0, 1.0, 2.0), record(10.1, 3.0, 4.0, valid=False)]
    path = tmp_path / "data" / "ghost.jsonl"
    ghost.save(recs, path)
    assert ghost.load(path) == recs


def test_trace_skips_invalid_rows_and_starts_at_record_zero():
    recs = [record(10.0, 0.0, 0.0, valid=False), record(10.5, 1.0, 2.0), record(11.0, 3.0, 4.0)]
    assert ghost.trace(recs) == [[0.5, 1.0, 2.0], [1.0, 3.0, 4.0]]
    assert ghost.trace([]) == []


def test_at_interpolates_between_points():
    tr = [[0.0, 0.0, 10.0], [1.0, 2.0, 20.0], [3.0, 6.0, 0.0]]
    assert ghost.at(tr, 0.0) == (0.0, 10.0)
    assert ghost.at(tr, 1.0) == (2.0, 20.0)
    assert ghost.at(tr, 3.0) == (6.0, 0.0)
    assert ghost.at(tr, 0.5) == (1.0, 15.0)
    assert ghost.at(tr, 2.0) == (4.0, 10.0)


def test_at_is_none_before_the_start_and_holds_the_end():
    tr = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert ghost.at(tr, -0.1) is None
    assert ghost.at(tr, 1.1) == (1.0, 1.0)
    assert ghost.at([], 0.0) is None
