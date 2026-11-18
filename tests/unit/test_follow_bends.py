import math

import pytest

from src.contracts import BallState, Intent
from src.control import ROLL
from src.io.settings import load
from src.modes import make_mode
from src.modes.path import apart, lengths, sight, tangent, turned

pytestmark = pytest.mark.unit

NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)
ROW = [(float(x), 0.0) for x in range(-100, 101, 10)]
BEND = [(float(x), 0.0) for x in range(-100, 1, 10)] + [(0.0, float(y)) for y in range(10, 101, 10)]
R = 50.0
ROUND = [(R * math.cos(a / 40), R * math.sin(a / 40)) for a in range(0, 120)]


def cfg_for(path, **control):
    cfg = load()
    cfg["rig"].update(ball_d_mm=16.0, plate_inner_mm=292.0)
    cfg["control"].update({"vel_ema_s": 0.0, **control}, path_mm=[list(p) for p in path])
    return cfg


def ball(x, y, vx=0.0, vy=0.0, valid=True, t=0.0):
    return BallState(x, y, vx, vy, valid=valid, t_s=t)


def led(cfg, spots):
    follow = make_mode("path", cfg)
    for x, y in spots:
        cmd = follow(ball(x, y), NO_INTENT, cfg)
    return follow, cmd


def pull(cmd, c, vx=0.0, vy=0.0):
    """The tilt toward the goal, the brake on the speed taken back out."""
    return cmd.tilt_x_deg + c["kd_deg_s_mm"] * vx, cmd.tilt_y_deg + c["kd_deg_s_mm"] * vy


def test_sight_runs_the_whole_lookahead_along_a_straight_and_on_past_its_end():
    cum = lengths(ROW)
    assert sight(ROW, cum, 0.0, 40.0, 3.0) == 40.0
    assert sight(ROW, cum, 185.0, 40.0, 3.0) == 40.0


def test_sight_stops_just_round_a_corner_where_the_line_leaves_the_straight():
    cum = lengths(BEND)
    # from 20 mm before the corner the chord to (0, d) passes the corner 20d/hypot(20, d) away
    d = 20 * 3 / math.sqrt(400 - 9)
    # and never past where the line leaves the straight
    assert 20 + d - 0.01 <= sight(BEND, cum, 80.0, 40.0, 3.0) <= 20 + d
    assert sight(BEND, cum, 40.0, 40.0, 3.0) == 40.0


def test_tangent_turned_and_apart():
    cum = lengths(BEND)
    assert tangent(BEND, cum, 0.0) == 0.0 and tangent(BEND, cum, 150.0) == pytest.approx(
        math.pi / 2
    )
    assert tangent(BEND, cum, 999.0) == tangent(BEND, cum, 105.0) == pytest.approx(math.pi / 2)
    assert turned(3.0, -3.0) == pytest.approx(2 * math.pi - 6.0)
    assert (
        apart((5.0, 4.0), (0.0, 0.0), (10.0, 0.0)) == 4.0
        and apart((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == 5.0
    )


def test_a_ball_that_cut_inside_a_corner_keeps_its_place_and_is_pulled_back_to_the_line():
    cfg = cfg_for(BEND)
    follow, _ = led(cfg, [(-100, 0), (-70, 0), (-30, 0)])
    assert follow.s == pytest.approx(70.0)
    cmd = follow(ball(-12, 14), NO_INTENT, cfg)
    assert follow.s == pytest.approx(70.0) and cmd.tilt_y_deg < 0.0
    follow(ball(-12, 5), NO_INTENT, cfg)
    assert follow.s == pytest.approx(88.0)


@pytest.mark.parametrize(
    "path, at, far",
    [(ROW, (-60, 0), 40.0), (BEND, (-20, 0), 20 + 20 * 3 / math.sqrt(391)), (BEND, (-5, 0), 15.0)],
    ids=["straight", "20 mm before a corner", "5 mm before it: never under path_capture_mm"],
)
def test_a_ball_at_bend_speed_is_pulled_only_as_far_as_the_line_runs_straight(path, at, far):
    cfg = cfg_for(path)
    c = cfg["control"]
    follow, _ = led(cfg, [(x, 0) for x in range(-100, at[0] + 1, 20)])
    cmd = follow(ball(*at, vx=c["path_bend_mm_s"]), NO_INTENT, cfg)
    assert math.hypot(*pull(cmd, c, vx=c["path_bend_mm_s"])) == pytest.approx(
        c["kp_deg_mm"] * far, rel=1e-3
    )
    cmd = follow(ball(*at, vx=c["path_bend_mm_s"] / 2), NO_INTENT, cfg)
    half = (far + c["path_lookahead_mm"]) / 2
    assert math.hypot(*pull(cmd, c, vx=c["path_bend_mm_s"] / 2)) == pytest.approx(
        c["kp_deg_mm"] * half, rel=1e-3
    )


@pytest.mark.parametrize("v", [30.0, 45.0, 60.0])
def test_a_ball_on_a_bend_is_pushed_inward_as_hard_as_its_speed_needs(v):
    cfg = cfg_for(ROUND)
    c = cfg["control"]
    follow, _ = led(cfg, ROUND[:9:4])
    x, y = ROUND[8]
    a = math.atan2(y, x)
    vx, vy = -v * math.sin(a), v * math.cos(a)
    px, py = pull(follow(ball(x, y, vx, vy), NO_INTENT, cfg), c, vx, vy)
    inward = -(px * math.cos(a) + py * math.sin(a))
    per_deg = ROLL * c["sim_g_mm_s2"] * math.pi / 180
    assert inward == pytest.approx(v * v / R / per_deg, rel=0.1)


def test_a_ball_held_beside_the_line_winds_a_tilt_back_across_it_only():
    cfg = cfg_for(ROW)
    c = cfg["control"]
    follow, first = led(cfg, [(-100, 0), (-70, 0), (-50, 0), (-50, 5)])
    for i in range(1, 201):
        cmd = follow(ball(-50, 5, t=i * 0.01), NO_INTENT, cfg)
    assert cmd.tilt_y_deg - first.tilt_y_deg == pytest.approx(-c["ki_deg_s_mm"] * 5 * 2.0, rel=0.02)
    assert cmd.tilt_x_deg == pytest.approx(first.tilt_x_deg)
    follow(ball(0, 0, valid=False, t=2.1), NO_INTENT, cfg)
    assert follow(ball(-50, 5, t=2.2), NO_INTENT, cfg) == first


def test_a_ball_beside_the_other_leg_of_a_hairpin_is_not_taken_to_be_there():
    legs = [(float(x), 0.0) for x in range(0, 101, 10)] + [
        (float(x), 10.0) for x in range(100, -1, -10)
    ]
    cfg = cfg_for(legs)
    follow, _ = led(cfg, [(0, 0), (40, 0), (80, 0)])
    assert follow.s == pytest.approx(80.0)
    follow(ball(92, 9.5), NO_INTENT, cfg)
    assert follow.s == pytest.approx(80.0)


def test_a_gap_in_the_heartbeat_drops_what_was_wound_up_against_a_lean():
    cfg = cfg_for(ROW)
    follow, first = led(cfg, [(-100, 0), (-70, 0), (-50, 0), (-50, 5)])
    for i in range(1, 201):
        wound = follow(ball(-50, 5, t=i * 0.01), NO_INTENT, cfg)
    # the loop calls resume() and then the mode on every tick without a heartbeat
    for i in range(201, 301):
        follow.resume()
        follow(ball(-50, 5, t=i * 0.01), NO_INTENT, cfg)
    after = follow(ball(-50, 5, t=3.01), NO_INTENT, cfg)
    assert wound.tilt_y_deg < first.tilt_y_deg - 0.15
    assert after.tilt_y_deg == pytest.approx(first.tilt_y_deg, abs=0.01)


@pytest.mark.parametrize("key", ["path_bend_mm_s", "kp_deg_mm"])
def test_a_zero_from_the_settings_page_does_not_stop_path_mode(key):
    cfg = cfg_for(BEND, **{key: 0})
    follow, _ = led(cfg, [(-100, 0), (-80, 0)])
    cmd = follow(ball(-60, 0, vx=40.0, t=0.01), NO_INTENT, cfg)
    assert math.isfinite(cmd.tilt_x_deg) and math.isfinite(cmd.tilt_y_deg)


def test_the_page_gets_how_far_along_the_ball_is_and_the_point_it_is_steered_for():
    cfg = cfg_for([*ROW, (100.0, 10.0), (100.0, 20.0)])
    follow = make_mode("path", cfg)
    follow(ball(-100, 30), NO_INTENT, cfg)
    assert follow.path_view()["chase"] == [-100, 0] and follow.path_view()["along_mm"] == 0
    for x in (-100, -70, -50):
        follow(ball(x, 0), NO_INTENT, cfg)
    assert follow.path_view()["along_mm"] == 50 and follow.path_view()["chase"] == [-10, 0]
    for x in range(-40, 90, 10):
        follow(ball(x, 0), NO_INTENT, cfg)
    # balance takes over with the line still turning ahead: the point is then the end
    assert follow.phase == "finish" and follow.path_view()["chase"] == [100, 20]
