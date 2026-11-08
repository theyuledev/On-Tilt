import pytest

from src.contracts import BallState, PlateCommand
from src.control.guard import Guard, need, stop_mm
from src.io.settings import load
from src.loop import Loop
from src.modes import MODES, Guarded, make_mode

pytestmark = pytest.mark.unit


def cfg(**control):
    c = load()
    c["rig"].update(us_per_deg_x=57.0, us_per_deg_y=57.0, limit_x_us=150, limit_y_us=150)
    c["rig"].update(startup_flag=True, startup_range_us=150, ball_d_mm=61.8)
    c["control"].update({"guard_on": True, **control})
    return c


def at(x, vx=0.0, t=0.0, valid=True):
    return BallState(x, 0.0, vx, 0.0, valid=valid, t_s=t)


def test_the_stop_rolls_on_through_the_latency_then_brakes():
    assert stop_mm(10.0, 100.0, 0.1, 500.0) == pytest.approx(30.0)
    assert stop_mm(-10.0, -100.0, 0.1, 500.0) == pytest.approx(-30.0)


def test_need_is_the_tilt_fraction_that_stops_the_ball_at_the_line():
    assert need(0.0, 100.0, 110.0, 0.1, 1000.0) == pytest.approx(0.05)
    assert need(50.0, -100.0, 110.0, 0.1, 1000.0) == 0.0
    # past the line the room bottoms out at 1 mm, so it asks for everything
    assert need(120.0, 100.0, 110.0, 0.1, 1000.0) > 1


def test_the_middle_of_the_plate_is_the_players_and_the_tilt_is_capped():
    g = Guard(cfg())
    assert g(at(0.0), PlateCommand(1.5, -1.5)) == PlateCommand(1.5, -1.5)
    assert g(at(0.0, t=0.01), PlateCommand(5.0, -5.0)) == PlateCommand(2.0, -2.0)
    assert g.guard_view() == {"w": [0.0, 0.0], "side": [0.0, 0.0]}


def test_w_engages_fast_and_hands_back_slowly():
    g = Guard(cfg())
    g(at(110.0, 200.0, t=0.0), PlateCommand(0.0, 0.0))
    g(at(110.0, 200.0, t=0.01), PlateCommand(0.0, 0.0))
    assert g.guard_w[0] == pytest.approx(0.01 / 0.03)
    assert g.guard_view()["side"][0] == 1.0
    w = g.guard_w[0]
    g(at(0.0, t=0.02), PlateCommand(0.0, 0.0))
    assert g.guard_w[0] == pytest.approx(w * (1 - 0.01 / 0.8))


def test_no_time_between_frames_leaves_w_alone():
    g = Guard(cfg())
    g(at(110.0, 200.0, t=0.0), PlateCommand(0.0, 0.0))
    g(at(110.0, 200.0, t=0.01), PlateCommand(0.0, 0.0))
    w = list(g.guard_w)
    g(at(110.0, 200.0, t=0.01), PlateCommand(0.0, 0.0))
    assert g.guard_w == w


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_the_guard_never_tilts_further_toward_the_rim_than_the_player(side):
    g = Guard(cfg())
    for k in range(20):
        player = side * (2.0 if k % 2 else -0.5)
        out = g(at(side * 105.0, side * 250.0, t=k * 0.01), PlateCommand(player, 0.0))
        assert side * out.tilt_x_deg <= side * player
    assert side * out.tilt_x_deg < 0  # braking by now


def test_a_player_already_steering_in_harder_keeps_his_command():
    g = Guard(cfg())
    for k in range(20):
        out = g(at(105.0, 5.0, t=k * 0.01), PlateCommand(-2.0, 0.0))
    assert out.tilt_x_deg == -2.0


def test_without_a_ball_only_the_cap_applies():
    g = Guard(cfg())
    assert g(at(110.0, 300.0, valid=False), PlateCommand(4.0, 0.0)) == PlateCommand(2.0, 0.0)


def test_only_the_hand_mode_is_guarded_and_guard_off_leaves_it_bare():
    wrapped = make_mode("hand", cfg())
    assert isinstance(wrapped, Guarded) and wrapped.mode is MODES["hand"]
    assert not isinstance(make_mode("balance", cfg()), Guarded)
    assert make_mode("assist", cfg()) is MODES["assist"]
    assert make_mode("hand", cfg(guard_on=False)) is MODES["hand"]


def test_the_committed_settings_ship_with_the_guard_off():
    assert make_mode("hand", load()) is MODES["hand"]
    assert Loop(load()).snapshot()["guard"] is None
