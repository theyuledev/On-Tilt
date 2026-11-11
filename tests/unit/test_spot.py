import math

import pytest

from src.contracts import BallState, Intent
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
