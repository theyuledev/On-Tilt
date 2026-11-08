import pytest

from src.contracts import BallState, PlateCommand
from src.control.guard import Guard, hold_line, need, stop_mm
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


def test_the_hold_line_eases_off_then_springs_back_but_never_past_the_cap():
    assert hold_line(50.0, 0.0, 100.0, 25.0, 2.0, 0.1, 0.02) == 2.0
    assert hold_line(100.0, 0.0, 100.0, 25.0, 2.0, 0.1, 0.02) == 0.0
    assert hold_line(110.0, 0.0, 100.0, 25.0, 2.0, 0.1, 0.02) == pytest.approx(-1.0)
    assert hold_line(110.0, 500.0, 100.0, 25.0, 2.0, 0.1, 0.02) == -2.0


def test_past_the_line_a_resting_player_is_pushed_in_and_inside_it_he_is_not():
    g = Guard(cfg())
    k = g.c["guard_hold_k_deg_mm"]
    assert g(at(g.edge + 5.0), PlateCommand(0.0, 0.0)).tilt_x_deg == pytest.approx(-5.0 * k)
    assert Guard(cfg())(at(g.edge + 5.0), PlateCommand(-1.0, 0.0)).tilt_x_deg == -1.0
    for player in (0.0, -1.0):
        g = Guard(cfg())
        assert g(at(g.edge - 5.0), PlateCommand(player, 0.0)).tilt_x_deg == player


@pytest.mark.parametrize("q", [10.0, 15.0])
@pytest.mark.parametrize("sp", [1.0, -1.0])
def test_the_hold_line_has_no_step_as_the_player_crosses_zero(q, sp):
    top = Guard(cfg()).cap[0]
    sweep = [-top + 2 * top * k / 400 for k in range(401)]
    out = []
    for player in sweep:
        g = Guard(cfg())
        out.append(g(at(sp * (g.edge + q)), PlateCommand(player, 0.0)).tilt_x_deg)
    for a, b, pa, pb in zip(out, out[1:], sweep, sweep[1:], strict=False):
        assert a - 1e-9 <= b <= a + (pb - pa) + 1e-9
    for o, player in zip(out, sweep, strict=True):
        assert sp * o <= sp * player + 1e-9 and -top <= o <= top


def test_an_outward_push_past_the_line_meets_a_spring():
    g = Guard(cfg())
    out = g(at(-(g.edge + 5.0)), PlateCommand(-2.0, 0.0)).tilt_x_deg
    assert out == pytest.approx(5.0 * g.c["guard_hold_k_deg_mm"])


def test_a_zero_hold_band_turns_the_hold_line_off():
    g = Guard(cfg(guard_hold_band_mm=0))
    assert g(at(g.edge + 5.0), PlateCommand(2.0, 0.0)).tilt_x_deg == 2.0


def test_a_ball_lost_while_braking_at_the_rim_fades_the_brake_out():
    g = Guard(cfg(guard_hold_band_mm=0))
    for k in range(20):
        braked = g(at(100.0, 200.0, t=0.01 * k), PlateCommand(2.0, 0.0)).tilt_x_deg
    assert braked < 1.0
    lost = [at(100.0, 200.0, t=0.2 + 0.01 * k, valid=False) for k in range(10)]
    xs = [g(b, PlateCommand(2.0, 0.0)).tilt_x_deg for b in lost]
    assert all(a <= b for a, b in zip([braked, *xs], xs, strict=False))
    assert xs[-1] < 2.0 - 0.8 * (2.0 - braked)
