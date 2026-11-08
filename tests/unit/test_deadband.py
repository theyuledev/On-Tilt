import pytest

from src.contracts import ServoCommand
from src.control.deadband import invert, start

pytestmark = pytest.mark.unit

RIG = {
    "center_x_us": 1500,
    "center_y_us": 1500,
    "trim_x_us": 0,
    "trim_y_us": 0,
    "limit_x_us": 150,
    "limit_y_us": 150,
    "startup_flag": False,
    "startup_range_us": 150,
}
CFG = {"rig": RIG, "control": {"servo_dead_band_us": 20, "db_comp_us": 20, "db_flip_us": 2}}


def drive(targets, state=None):
    state, out = state or start(RIG), []
    for x in targets:
        state, cmd = invert(state, ServoCommand(x, 1500), False, CFG)
        out.append(cmd.x_us)
    return state, out


def test_jitter_inside_the_hysteresis_never_moves_anything():
    assert drive([1500, 1501, 1502, 1501])[1] == [1500, 1500, 1500, 1500]


def test_a_real_move_leads_by_half_the_band_then_holds_inside_it():
    assert drive([1510, 1515, 1529, 1530])[1] == [1520, 1520, 1520, 1540]


def test_turning_round_past_the_hysteresis_jumps_a_whole_band():
    assert drive([1530, 1528, 1527])[1] == [1540, 1540, 1517]


def test_unchanged_target_keeps_its_heading():
    assert drive([1530, 1530, 1530])[1] == [1540, 1540, 1540]


def test_killing_or_disarmed_passes_through_and_forgets_its_place():
    state, _ = drive([1530])
    state, cmd = invert(state, ServoCommand(1505, 1500), True, CFG)
    assert cmd == ServoCommand(1505, 1500) and state == start(RIG)
    assert invert(state, None, False, CFG) == (start(RIG), None)


def test_lead_never_goes_past_the_limits():
    assert drive([1645])[1] == [1650]


def test_axes_are_independent():
    state, cmd = invert(start(RIG), ServoCommand(1500, 1470), False, CFG)
    assert (cmd.x_us, cmd.y_us) == (1500, 1460)
