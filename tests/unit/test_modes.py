import pytest

from src.contracts import BallState, Intent
from src.modes import make_mode

pytestmark = pytest.mark.unit

CFG = {
    "rig": {"plate_inner_mm": 300.0},
    "control": {
        "kp_deg_mm": 0.1,
        "kd_deg_s_mm": 0.01,
        "d_max_deg": 2.0,
        "ki_deg_s_mm": 0.015,
        "i_max_deg": 1.0,
        "i_err_max_mm": 30,
        "vel_ema_s": 0.0,
        "balance_reach_mm": 500,
        "balance_rest_mm": 0,
        "balance_rest_mm_s": 10,
        "balance_rest_out_mm": 8,
        "balance_rest_trim": 1.0,
        "balance_rest_grow": 0.0,
        "balance_rest_still_mm_s": 0,
        "balance_rest_still_s": 0.3,
        "balance_rest_move_mm_s": 12,
        "balance_rest_soft": False,
        "balance_rest_edge_mm": 25,
    },
}
NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)


def tilt(x, y, vx=0.0, vy=0.0, valid=True):
    ball = BallState(x, y, vx, vy, valid=valid, t_s=0.0)
    cmd = make_mode("balance", CFG)(ball, NO_INTENT, CFG)
    return pytest.approx((cmd.tilt_x_deg, cmd.tilt_y_deg))


def test_balance_tilts_back_toward_centre():
    assert tilt(10.0, -20.0) == (-1.0, 2.0)


def test_balance_tilts_against_motion():
    assert tilt(0.0, 0.0, 50.0, -100.0) == (-0.5, 1.0)


def test_balance_clamps_the_d_term():
    assert tilt(10.0, 0.0, 1000.0, -1000.0) == (-3.0, 2.0)


def test_balance_levels_without_a_ball():
    assert tilt(50.0, 50.0, valid=False) == (0.0, 0.0)
