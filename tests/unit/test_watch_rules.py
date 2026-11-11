import pytest

from src.io.settings import load
from src.safety.watch import problems, swings

pytestmark = pytest.mark.unit

CFG = load()
RIG, C = CFG["rig"], CFG["control"]
DT = 0.05


def rows(n, **over):
    """n rows 50 ms apart; a value may be a function of the row number."""
    base = {
        "armed": True,
        "auto": True,
        "seen": True,
        "pos": (0.0, 0.0),
        "vel": (0.0, 0.0),
        "us": (0, 0),
        "deg": (0.0, 0.0),
        "board_lost": False,
        "camera_lost": False,
    }
    return [
        {**base, "t": i * DT, **{k: v(i) if callable(v) else v for k, v in over.items()}}
        for i in range(n)
    ]


def check(r):
    return problems(r, RIG, C)


def test_a_calm_armed_plate_and_a_disarmed_one_report_nothing():
    assert check(rows(100)) == []
    assert check(rows(100, armed=False, seen=False)) == []


def test_small_chatter_is_not_a_swing_and_big_turns_are_counted():
    assert swings([0, 20, -20, 20, -20, 20], 100) == 0
    assert swings([0, 120, -120, 120, -120], 100) == 3


def test_ball_gone_is_said_only_after_the_blind_time_and_only_while_armed():
    short = round(C["watch_blind_s"] / DT) - 2
    assert check(rows(60, seen=lambda i: i < 60 - short)) == []
    assert check(rows(60, seen=lambda i: i < 20)) == ["ball not seen"]
    assert check(rows(60, seen=False, armed=False)) == []


def test_full_swings_several_times_a_second_trip_and_todays_worst_does_not():
    wild = rows(80, us=lambda i: (140 if (i // 4) % 2 else -140, 0))
    assert "plate is swinging hard" in check(wild)
    # nine big turns in three seconds was the most seen in a normal session
    busy = rows(80, us=lambda i: (120 if (i // 7) % 2 else -120, 0))
    assert "plate is swinging hard" not in check(busy)


def test_a_plate_parked_at_its_limit_trips_after_the_pinned_time():
    lim = RIG["limit_x_us"]
    n = round(C["watch_pinned_s"] / DT)
    assert "plate held at its limit" not in check(
        rows(100, us=lambda i: (lim if i > 100 - n + 4 else 0, 0))
    )
    assert "plate held at its limit" in check(
        rows(100, us=lambda i: (lim if i > 100 - n - 4 else 0, 0))
    )


def test_a_steady_tilt_the_ball_ignores_trips_and_one_it_follows_does_not():
    tilt = 2 * C["watch_deaf_deg"]
    assert "ball is not answering the plate" in check(rows(100, deg=(tilt, 0.0)))
    # 150 mm/s gained in the span is a ball that answers
    moving = rows(100, deg=(tilt, 0.0), vel=lambda i: (max(0, i - 60) * 5.0, 0.0))
    assert "ball is not answering the plate" not in check(moving)
    # by the wall, or with the ball unseen, there is nothing to judge
    assert "ball is not answering the plate" not in check(
        rows(100, deg=(tilt, 0.0), pos=(RIG["plate_inner_mm"] / 2 - 5, 0.0))
    )


def test_link_and_camera_loss_are_said_armed_or_not():
    assert check(rows(10, armed=False, board_lost=True)) == ["board link lost"]
    assert "camera lost" in check(rows(10, camera_lost=True))


def test_a_player_may_hold_full_tilt_and_a_mode_steering_on_the_ball_may_not():
    held = {"us": (RIG["limit_x_us"], 0), "deg": (2.0, 0.0)}
    assert check(rows(80, auto=False, **held)) == []
    assert check(rows(80, **held)) == ["plate held at its limit", "ball is not answering the plate"]
