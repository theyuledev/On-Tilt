import math

import pytest

from src.contracts import BallState, Intent
from src.io.settings import load
from src.modes import Balance, make_mode
from src.modes.spot import aim

pytestmark = pytest.mark.unit

NO_INTENT = Intent(0.0, 0.0, valid=False, t_s=0.0)
ROW = [(float(x), 0.0) for x in range(-100, 101, 10)]
BEND = [(float(x), 0.0) for x in range(-100, 1, 10)] + [(0.0, float(y)) for y in range(10, 101, 10)]
# a C whose two ends lie 10 mm apart
HOOK = [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)]
ALONG_ROW = [(-100, 0), (-70, 0), (-40, 0), (-10, 0), (20, 0), (50, 0)]


def cfg_for(path, **control):
    cfg = load()
    cfg["rig"].update(ball_d_mm=16.0, plate_inner_mm=292.0)
    cfg["control"].update(control, path_mm=[list(p) for p in path])
    return cfg


def ball(x, y, vx=0.0, vy=0.0, valid=True, t=0.0):
    return BallState(x, y, vx, vy, valid=valid, t_s=t)


def full(cfg):
    return cfg["control"]["kp_deg_mm"] * cfg["control"]["path_lookahead_mm"]


def pull_deg(cmd):
    return math.hypot(cmd.tilt_x_deg, cmd.tilt_y_deg)


def led(cfg, spots):
    """A follower that has seen a resting ball at each spot in turn, and its last tilt."""
    follow = make_mode("path", cfg)
    for x, y in spots:
        cmd = follow(ball(x, y), NO_INTENT, cfg)
    return follow, cmd


def test_open_path_starts_at_the_end_nearest_the_ball():
    cfg = cfg_for(ROW)
    follow = make_mode("path", cfg)
    assert follow.path_view() == {"done": False, "points": [list(p) for p in ROW]}
    follow(ball(95, 3), NO_INTENT, cfg)
    assert follow.path_view()["points"] == [list(p) for p in ROW[::-1]]
    assert follow(ball(95, 3), NO_INTENT, cfg).tilt_x_deg < -0.95 * full(cfg)


def test_open_path_goes_to_its_start_before_it_follows():
    cfg = cfg_for(ROW)
    follow, cmd = led(cfg, [(-100, 60)])
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((0.0, -full(cfg)))
    assert follow.open.phase == "join"
    assert follow(ball(-100, 10), NO_INTENT, cfg).tilt_x_deg > 0.95 * full(cfg)
    assert follow.open.phase == "follow"


@pytest.mark.parametrize(
    "path, spots",
    [
        (ROW, [(-100, 25)]),
        (ROW, ALONG_ROW[:4]),
        (BEND, [(-100, 0), (-70, 0), (-40, 0), (-20, 0)]),
        (ROW, [*ALONG_ROW, (65, 0)]),
    ],
    ids=["beside the start", "on a straight", "20 mm before a corner", "35 mm before the end"],
)
def test_a_ball_at_rest_always_gets_the_full_pull(path, spots):
    cfg = cfg_for(path)
    _, cmd = led(cfg, spots)
    assert pull_deg(cmd) == pytest.approx(full(cfg))


def test_open_path_levels_blind_keeps_its_place_and_picks_up_where_the_ball_returns():
    cfg = cfg_for(ROW)
    follow, _ = led(cfg, [(-100, 0), (-70, 0), (-50, 0)])
    cmd = follow(ball(999, 999, valid=False), NO_INTENT, cfg)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == (0.0, 0.0)
    assert follow.open.s == pytest.approx(50.0) and follow.open.phase == "follow"
    # 110 mm further on: it catches up a lookahead a tick, so a later pass is never jumped to
    cmd = follow(ball(60, 4), NO_INTENT, cfg)
    assert follow.open.s == pytest.approx(90.0) and pull_deg(cmd) == pytest.approx(full(cfg))
    for _ in range(2):
        cmd = follow(ball(60, 4), NO_INTENT, cfg)
    assert follow.open.s == pytest.approx(160.0) and cmd.tilt_x_deg > 0.95 * full(cfg)


def test_balance_takes_over_inside_i_err_max_mm_of_the_last_point_and_winds_its_integral():
    cfg = cfg_for(ROW)
    follow, before = led(cfg, [*ALONG_ROW, (69, 0)])
    assert follow.open.phase == "follow"
    first = follow(ball(71, 0, t=1.0), NO_INTENT, cfg)
    assert follow.open.phase == "finish"
    assert first.tilt_x_deg == pytest.approx(cfg["control"]["kp_deg_mm"] * 29)
    assert before.tilt_x_deg - first.tilt_x_deg < 0.6
    later = follow(ball(71, 0, t=2.0), NO_INTENT, cfg)
    assert later.tilt_x_deg > first.tilt_x_deg + 0.5


def test_the_hand_over_to_balance_keeps_the_smoothed_speed():
    cfg = cfg_for(ROW, vel_ema_s=0.1)
    c = cfg["control"]
    follow = make_mode("path", cfg)
    for i, (x, y) in enumerate([*ALONG_ROW, (69, 0)]):
        follow(ball(x, y, vx=40.0, t=i * 0.01), NO_INTENT, cfg)
    cmd = follow(ball(71, 0, vx=400.0, t=0.07), NO_INTENT, cfg)
    assert cmd.tilt_x_deg == pytest.approx(c["kp_deg_mm"] * 29 - c["kd_deg_s_mm"] * 40.0)


def test_a_hand_over_straight_after_a_blind_stretch_brakes_on_the_fresh_speed():
    cfg = cfg_for(ROW, vel_ema_s=0.1)
    follow = make_mode("path", cfg)
    for i, (x, y) in enumerate([*ALONG_ROW, (69, 0)]):
        follow(ball(x, y, vx=40.0, t=i * 0.01), NO_INTENT, cfg)
    follow(ball(0, 0, valid=False, t=0.5), NO_INTENT, cfg)
    cmd = follow(ball(71, 0, t=1.0), NO_INTENT, cfg)
    assert cmd.tilt_x_deg == pytest.approx(cfg["control"]["kp_deg_mm"] * 29)


def test_follow_brakes_on_the_raw_speed_unless_vel_ema_s_smooths_it():
    tilts = {}
    for ema in (0.0, 0.1):
        cfg = cfg_for(ROW, vel_ema_s=ema)
        follow = make_mode("path", cfg)
        for i, (x, vx) in enumerate([(-100, 0.0), (-70, 0.0), (-40, 300.0)]):
            cmd = follow(ball(x, 0, vx=vx, t=i * 0.01), NO_INTENT, cfg)
        tilts[ema] = cmd.tilt_x_deg
    c = cfg["control"]
    assert tilts[0.0] == pytest.approx(full(cfg) - c["d_max_deg"])
    assert tilts[0.1] == pytest.approx(full(cfg) - c["kd_deg_s_mm"] * 30.0)


def test_done_latches_near_the_last_point_only_once_the_path_has_been_followed():
    cfg = cfg_for(HOOK)
    follow, _ = led(cfg, [(0, 4)])
    assert follow.path_view()["points"][0] == [0.0, 0.0] and not follow.done
    for spot in [(30, 0), (60, 0), (90, 0), (100, 5), (80, 10), (50, 10), (25, 10)]:
        follow(ball(*spot), NO_INTENT, cfg)
    assert follow.open.phase == "finish" and not follow.done
    follow(ball(2, 10, valid=False), NO_INTENT, cfg)
    assert not follow.done
    follow(ball(12, 10), NO_INTENT, cfg)
    assert follow.done
    follow(ball(0, 0, valid=False), NO_INTENT, cfg)
    follow(ball(90, 90), NO_INTENT, cfg)
    assert follow.done and follow.path_view()["done"]


def test_points_beyond_the_reach_are_pulled_in_and_repeats_dropped():
    cfg = cfg_for([(0, 0), (100, 0), (138, 0), (150, 0), (150, 50)])
    points = make_mode("path", cfg).path_view()["points"]
    assert points == [[0, 0], [100, 0], [118, 0], [118, 50]]


def test_the_page_gets_the_points_in_whole_mm():
    cfg = cfg_for([(0.4, 0.6), (100.4, -0.6)])
    points = make_mode("path", cfg).path_view()["points"]
    assert points == [[0, 1], [100, -1]] and all(type(v) is int for p in points for v in p)


def test_balance_toward_pulls_to_the_goal_it_is_given_not_the_held_target():
    cfg = cfg_for(ROW)
    cmd = Balance(cfg).toward(ball(0, 0), (20.0, 0.0), cfg)
    assert (cmd.tilt_x_deg, cmd.tilt_y_deg) == pytest.approx((cfg["control"]["kp_deg_mm"] * 20, 0))
    aim(cfg, [20.0, 0.0])
    assert Balance(cfg)(ball(0, 0), NO_INTENT, cfg) == cmd
