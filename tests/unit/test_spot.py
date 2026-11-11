import math

import pytest

from src.contracts import BallState, Intent
from src.control.pid import Integral
from src.io.settings import load
from src.loop import Loop
from src.modes import make_mode
from src.modes.assist import reach_mm
from src.modes.spot import CENTRE, aim, held

pytestmark = pytest.mark.unit

NO_INTENT = Intent(0.0, 0.0, False, 0.0)


@pytest.fixture
def cfg():
    return load()


def test_balance_holds_the_centre_until_a_target_is_set(cfg):
    assert held(cfg) == CENTRE and CENTRE["mm"] == (0.0, 0.0)
    assert Loop(cfg).snapshot()["target"] == CENTRE


def test_a_click_becomes_the_target_and_is_kept_off_the_rim(cfg):
    reach = reach_mm(cfg["rig"], cfg["control"]["assist_margin_mm"])
    aim(cfg, [60, -45.5])
    assert held(cfg) == {"src": "mouse", "mm": (60.0, -45.5), "found": True}
    aim(cfg, [1000, -1000])
    assert held(cfg)["mm"] == (reach, -reach)
    assert 0 < reach < cfg["rig"]["plate_inner_mm"] / 2 - cfg["rig"]["ball_d_mm"] / 2


@pytest.mark.parametrize("bad", [[1], [1, 2, 3], ["a", 2], [math.nan, 0], [0, math.inf], {}, 5])
def test_a_click_that_is_not_a_point_changes_nothing(cfg, bad):
    aim(cfg, [10, 20])
    with pytest.raises((ValueError, TypeError)):
        aim(cfg, bad)
    assert held(cfg)["mm"] == (10.0, 20.0)


def test_mouse_keeps_the_point_centre_resets_it(cfg):
    aim(cfg, [30, 40])
    aim(cfg, "marker")
    aim(cfg, "mouse")
    assert held(cfg) == {"src": "mouse", "mm": (30.0, 40.0), "found": True}
    aim(cfg, "centre")
    assert held(cfg) == CENTRE


def test_a_found_x_moves_the_target_and_a_missing_one_leaves_it(cfg):
    aim(cfg, [30, 40])
    aim(cfg, "marker")
    assert held(cfg) == {"src": "marker", "mm": (30.0, 40.0), "found": False}
    aim(cfg, "marker", (-60.0, 45.0))
    assert held(cfg) == {"src": "marker", "mm": (-60.0, 45.0), "found": True}
    aim(cfg, "marker", (-9999.0, 0.0))
    assert held(cfg)["mm"][0] == -reach_mm(cfg["rig"], cfg["control"]["assist_margin_mm"])


def test_balance_tilts_about_the_target_exactly_as_it_did_about_the_centre(cfg):
    at_centre = make_mode("balance", cfg)(
        BallState(12.0, -7.0, 30.0, 5.0, True, 1.0), NO_INTENT, cfg
    )
    aim(cfg, [50, 20])
    moved = BallState(62.0, 13.0, 30.0, 5.0, True, 1.0)
    assert make_mode("balance", cfg)(moved, NO_INTENT, cfg) == at_centre
    on_it = make_mode("balance", cfg)(BallState(50.0, 20.0, 0.0, 0.0, True, 1.0), NO_INTENT, cfg)
    assert (on_it.tilt_x_deg, on_it.tilt_y_deg) == (0.0, 0.0)


def test_a_ball_past_the_reach_is_pulled_no_harder_than_one_at_it(cfg):
    c = cfg["control"]
    reach = c["balance_reach_mm"]

    def tilt(x, y):
        return make_mode("balance", cfg)(BallState(x, y, 0.0, 0.0, True, 0.0), NO_INTENT, cfg)

    side = reach / math.sqrt(2)
    assert tilt(3 * reach, 0.0) == tilt(reach, 0.0)
    assert tilt(4 * side, 4 * side).tilt_x_deg == pytest.approx(tilt(side, side).tilt_x_deg)
    assert abs(tilt(reach / 2, 0.0).tilt_x_deg) < abs(tilt(reach, 0.0).tilt_x_deg)
    # the pull stays inside what the brake term may ask for, or a far ball never stops speeding up
    assert c["kp_deg_mm"] * reach <= c["d_max_deg"]


def resting(cfg, steps):
    """Tilts from one balance mode fed (x_mm, speed) steps 10 ms apart, with the rest zone on."""
    cfg["control"] |= {"balance_rest_mm": 5, "balance_rest_mm_s": 10, "balance_rest_out_mm": 8}
    mode = make_mode("balance", cfg)
    return [
        mode(BallState(x, 0.0, v, 0.0, True, i * 0.01), NO_INTENT, cfg).tilt_x_deg
        for i, (x, v) in enumerate(steps)
    ]


def test_a_ball_close_and_slow_is_left_alone_until_it_drifts_past_the_wake_line(cfg):
    still, drifted, awake = resting(cfg, [(3.0, 0.0), (7.0, 0.0), (9.0, 0.0)])
    assert still == drifted == 0.0 and awake < 0.0


def test_a_ball_moving_inside_the_rest_zone_is_braked_but_not_pulled(cfg):
    c = cfg["control"]
    _, moving = resting(cfg, [(3.0, 0.0), (3.0, 6.0)])
    assert moving == pytest.approx(-c["kd_deg_s_mm"] * 6.0)


def test_a_close_ball_that_is_still_moving_fast_is_not_left_alone(cfg):
    assert resting(cfg, [(3.0, 50.0)])[0] < 0.0


def test_the_trim_does_not_wind_while_the_ball_rests_off_the_spot(cfg):
    assert resting(cfg, [(3.0, 0.0)] * 300)[-1] == 0.0


def test_the_rest_zone_is_off_until_it_is_given_a_size(cfg):
    tilt = make_mode("balance", cfg)(BallState(3.0, 0.0, 0.0, 0.0, True, 0.0), NO_INTENT, cfg)
    assert tilt.tilt_x_deg < 0.0


def test_a_ball_rolling_through_is_not_caught_because_its_smoothed_speed_lags(cfg):
    cfg["control"]["vel_ema_s"] = 0.1
    assert resting(cfg, [(20.0, 0.0), (3.0, 30.0)])[1] < 0.0


def test_a_resting_plate_holds_only_the_share_of_its_trim_it_is_told_to(cfg):
    cfg["control"] |= {"balance_rest_mm": 5, "balance_rest_mm_s": 10, "balance_rest_trim": 0.5}
    mode = make_mode("balance", cfg)
    mode.acc = Integral(0.4, -0.2, 0.0)
    tilt = mode(BallState(3.0, 0.0, 0.0, 0.0, True, 0.01), NO_INTENT, cfg)
    assert (tilt.tilt_x_deg, tilt.tilt_y_deg) == pytest.approx((-0.2, 0.1))


def test_the_rest_zone_is_wider_round_a_target_far_from_the_middle(cfg):
    cfg["control"] |= {"balance_rest_mm": 10, "balance_rest_mm_s": 1000, "balance_rest_out_mm": 14}
    cfg["control"] |= {"balance_rest_trim": 0.0, "balance_rest_grow": 0.1}

    def tilt(target_x, off):
        aim(cfg, [target_x, 0.0])
        ball = BallState(target_x + off, 0.0, 0.0, 0.0, True, 0.0)
        return make_mode("balance", cfg)(ball, NO_INTENT, cfg).tilt_x_deg

    assert tilt(0.0, 12.0) < 0.0  # 12 mm from a target in the middle: outside its 10 mm zone
    assert tilt(100.0, 12.0) == 0.0  # the same 12 mm, target 100 mm out: inside its 20 mm zone
    assert tilt(100.0, 22.0) < 0.0


def test_the_brake_lets_go_of_a_ball_that_has_lain_still_and_comes_back_when_it_moves(cfg):
    cfg["control"] |= {"balance_rest_still_mm_s": 4, "balance_rest_still_s": 0.3}
    cfg["control"] |= {"balance_rest_move_mm_s": 12}
    tilts = resting(cfg, [(3.0, 2.0)] * 40 + [(3.0, 8.0), (3.0, 20.0)])
    c = cfg["control"]
    assert tilts[10] == pytest.approx(-c["kd_deg_s_mm"] * 2.0)  # still braking in the first 0.3 s
    assert tilts[39] == 0.0 and tilts[40] == 0.0  # let go, and a small stir does not bring it back
    assert tilts[41] == pytest.approx(-c["kd_deg_s_mm"] * 20.0)


def test_soft_rest_drops_the_trim_on_the_way_in_and_wakes_with_a_nudge(cfg):
    cfg["control"] |= {"balance_rest_mm": 10, "balance_rest_mm_s": 1000, "balance_rest_out_mm": 14}
    cfg["control"] |= {"balance_rest_trim": 0.0, "balance_rest_soft": True}
    mode = make_mode("balance", cfg)
    mode.acc = Integral(0.4, -0.2, 0.0)
    mode(BallState(3.0, 0.0, 0.0, 0.0, True, 0.01), NO_INTENT, cfg)
    assert (mode.acc.x, mode.acc.y) == (0.0, 0.0)
    woken = make_mode("balance", cfg)(BallState(14.0, 0.0, 0.0, 0.0, True, 0.0), NO_INTENT, cfg)
    assert woken.tilt_x_deg == pytest.approx(-cfg["control"]["kp_deg_mm"] * 4.0)


def test_a_ball_near_the_edge_of_the_board_is_pulled_in_even_inside_the_zone(cfg):
    cfg["control"] |= {"balance_rest_mm": 10, "balance_rest_mm_s": 1000, "balance_rest_out_mm": 30}
    aim(cfg, [1000.0, 0.0])
    tx = held(cfg)["mm"][0]
    # the edge rule starts 2 mm outside the target: 4 mm inside is clear of it, 8 mm out is not
    cfg["control"] |= {"balance_rest_edge_mm": cfg["rig"]["plate_inner_mm"] / 2 - tx - 2}

    def tilt(x):
        ball = BallState(x, 0.0, 0.0, 0.0, True, 0.0)
        return make_mode("balance", cfg)(ball, NO_INTENT, cfg).tilt_x_deg

    assert tilt(tx - 4.0) == 0.0
    assert tilt(tx + 8.0) < 0.0
