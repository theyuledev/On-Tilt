import math
import random
import time

import pytest

from src.app import Loop
from src.contracts import BallState, Intent
from src.io.settings import load
from src.modes.path import target
from src.sim.ball import roll
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

BALLS = (61.8, 38.0, 13.0)
CONFIGS = ((57.0, 150), (57.0, 300), (28.75, 150))
PUSHES = {
    "push +x": lambda t: (1.0, 0.0),
    "push corner": lambda t: (1.0, 1.0),
    "push then let go": lambda t: (1.0 if t < 0.6 else 0.0, 0.0),
    "panic sawtooth": lambda t: (1.0 if int(t) % 2 == 0 else -1.0, 0.0),
}
SQ = [(100, 100), (-100, 100), (-100, -100), (100, -100), (100, 100)]


class Hand:
    seen = calibrating = False
    progress = 0.0

    def __init__(self):
        self.intent = Intent(0.0, 0.0, valid=False, t_s=0.0)

    def latest(self) -> Intent:
        return self.intent


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def rig_cfg(ball_d=61.8, lim=150, upd=57.0, cap=None):
    cfg = load()
    rig = cfg["rig"]
    rig.update(ball_d_mm=ball_d, limit_x_us=lim, limit_y_us=lim, startup_range_us=150)
    rig.update(startup_flag=lim <= 150, us_per_deg_x=upd, us_per_deg_y=upd)
    # no rolling friction on purpose: friction helps the brake, so this is the worst case
    cfg["control"].update(rolling_friction_x_mm_s2=0.0, rolling_friction_y_mm_s2=0.0)
    if cap is not None:
        cfg["control"]["player_max_deg"] = cap
    return cfg


def linear_hand(cfg):
    # intent straight through as degrees, for players that steer in degrees
    cfg["input"].update(dead_zone=0.0, expo=0.0, intent_gain_deg=1.0)
    return cfg


def run(cfg, clock, player, seconds=5.0, start=(0.0, 0.0), keys=False, plant=None):
    """Hand mode on a sim plate; the guard sees the ball late, noisy, with a 4 frame velocity."""
    plant = plant or cfg
    hand = Hand()
    loop = Loop(
        cfg, hand=None if keys else hand, actuator=SimActuator(plant["rig"], plant["control"])
    )
    loop.arm()
    dt = 1 / cfg["control"]["sim_tick_hz"]
    lag = math.ceil(cfg["vision"]["camera_latency_s"] / dt - 1e-9)
    wall = (plant["rig"]["plate_inner_mm"] - plant["rig"]["ball_d_mm"]) / 2
    rng = random.Random(1)
    t0 = clock[0]
    ball = BallState(*start, 0.0, 0.0, valid=True, t_s=t0)
    seen, out = [], {"gap": wall, "corr": 0, "track": [], "pulses": []}
    for _ in range(round(seconds / dt)):
        clock[0] += dt
        seen.append((ball.t_s, ball.x_mm + rng.gauss(0, 0.15), ball.y_mm + rng.gauss(0, 0.15)))
        t, x, y = seen[max(len(seen) - 1 - lag, 0)]
        ta, xa, ya = seen[max(len(seen) - 4 - lag, 0)]
        span = max(t - ta, dt)
        loop.ball = BallState(x, y, (x - xa) / span, (y - ya) / span, valid=True, t_s=t)
        ix, iy = player(clock[0] - t0, loop.ball)
        if keys:
            loop.keys.set(ix, iy)
        else:
            hand.intent = Intent(ix, iy, valid=True, t_s=clock[0])
        rec = loop.tick(clock[0], dt)
        asked = loop.step.mode(loop.ball, rec.intent, cfg)
        capped = [
            min(max(a, -c), c)
            for a, c in zip((asked.tilt_x_deg, asked.tilt_y_deg), loop.step.cap, strict=True)
        ]
        got = (rec.plate_cmd.tilt_x_deg, rec.plate_cmd.tilt_y_deg)
        out["corr"] += max(abs(a - b) for a, b in zip(got, capped, strict=True)) > 0.05
        ball = roll(ball, rec.plate_state, clock[0], plant["rig"], plant["control"])
        out["gap"] = min(out["gap"], wall - max(abs(ball.x_mm), abs(ball.y_mm)))
        out["track"].append((ball.x_mm, ball.y_mm))
        out["pulses"].append(rec.servo_cmd.x_us)
    out["corr"] /= round(seconds / dt)
    return out


def worst_push(cfg, clock, keys=False, plant=None):
    pushes = PUSHES.values()
    return min(
        run(cfg, clock, lambda t, b, f=f: f(t), keys=keys, plant=plant)["gap"] for f in pushes
    )


@pytest.mark.parametrize("keys", [False, True], ids=["hand", "keys"])
@pytest.mark.parametrize("ball_d", BALLS)
@pytest.mark.parametrize("upd,lim", CONFIGS)
def test_a_full_tilt_push_stops_short_of_the_rim(clock, upd, lim, ball_d, keys):
    assert worst_push(rig_cfg(ball_d, lim, upd), clock, keys) > 5


@pytest.mark.xfail(reason="without the player cap a 5 deg push outruns the brake")
@pytest.mark.parametrize("ball_d", BALLS)
def test_an_uncapped_push_at_the_wide_clamp_reaches_the_rim(clock, ball_d):
    assert worst_push(rig_cfg(ball_d, 300, 57.0, cap=99.0), clock) > 5


def pd(goal, kp=0.05, kd=0.0375, most=2.0):
    def f(b):
        d = [
            kp * (g - p) - kd * v
            for g, p, v in zip(goal, (b.x_mm, b.y_mm), (b.vx_mm_s, b.vy_mm_s), strict=True)
        ]
        return tuple(min(max(a, -most), most) for a in d)

    return f


def lap(ahead=40.0, **gains):
    seg = [0]

    def f(t, b):
        seg[0], goal = target(SQ, seg[0], (b.x_mm, b.y_mm), ahead, 15)
        return pd(goal, **gains)(b)

    return f, seg


def corners_hit(track, ball_d):
    left = list(SQ[1:])
    for p in track:
        if left and math.dist(p, left[0]) < ball_d / 2 + 15:
            left.pop(0)
    return not left


@pytest.mark.parametrize("ball_d", BALLS)
def test_a_lap_of_the_square_maze_is_barely_touched(clock, ball_d):
    cfg = linear_hand(rig_cfg(ball_d))
    player, _ = lap()
    r = run(cfg, clock, player, seconds=20.0, start=(100.0, 100.0))
    assert corners_hit(r["track"], ball_d) and r["corr"] < 0.10


@pytest.mark.parametrize("ball_d", BALLS)
def test_holding_the_start_ring_does_not_drift(clock, ball_d):
    hold = pd((100.0, 100.0))
    cfg = linear_hand(rig_cfg(ball_d))
    r = run(cfg, clock, lambda t, b: hold(b), seconds=4.0, start=(100.0, 100.0))
    assert max(math.dist(p, (100, 100)) for p in r["track"][50:]) < 5


@pytest.mark.parametrize("ball_d", BALLS)
def test_a_hard_player_overshooting_the_corners_never_reaches_the_rim(clock, ball_d):
    cfg = linear_hand(rig_cfg(ball_d))
    player, _ = lap(ahead=80.0, kp=0.1, kd=0.0375)
    assert run(cfg, clock, player, seconds=10.0, start=(100.0, 100.0))["gap"] > 5
    hold = pd((100.0, 100.0), kp=0.1)
    # starts 20 mm short of the goal so the overshoot heads for the rim
    r = run(cfg, clock, lambda t, b: hold(b), seconds=3.0, start=(80.0, 100.0))
    assert r["gap"] > 5 and math.dist(r["track"][-1], (100, 100)) < 30


def reversals(pulses, band):
    # a turn counts once it has come back past the servo dead band, smaller ones never move it
    n, ext, up = 0, pulses[0], None
    for p in pulses:
        if up is None:
            if abs(p - ext) >= band:
                up, ext = p > ext, p
        elif (p > ext) == up:
            ext = p
        elif abs(p - ext) >= band:
            n, up, ext = n + 1, not up, p
    return n


def test_the_guard_holding_a_pusher_does_not_chatter(clock):
    cfg = rig_cfg()
    pulses = run(cfg, clock, lambda t, b: (1.0, 0.0), seconds=8.0)["pulses"][200:]
    assert reversals(pulses, cfg["control"]["servo_dead_band_us"]) / 6.0 < 2


@pytest.mark.parametrize("lim", [150, 300])
@pytest.mark.parametrize("ball_d", BALLS)
def test_a_slow_servo_chain_still_stops_the_push(clock, ball_d, lim):
    cfg = rig_cfg(ball_d, lim, 100.0)
    cfg["control"]["servo_dead_time_s"] = 0.08
    assert worst_push(cfg, clock) > 5


@pytest.mark.parametrize("lim,gap", [(150, 0), (300, 5)])
@pytest.mark.parametrize("ball_d", BALLS)
def test_a_guard_that_overrates_the_linkage_still_stops_the_push(clock, ball_d, lim, gap):
    # it thinks 57 us/deg, the plate really needs 100: both the push and the brake come out weaker.
    # at +-150 a 1.75x wrong gain only gets no rim hit, not the 5 mm margin
    truth = rig_cfg(ball_d, lim, 100.0)
    assert worst_push(rig_cfg(ball_d, lim, 57.0), clock, plant=truth) > gap


def test_the_page_sees_the_guard_weight_and_side(clock):
    loop = Loop(rig_cfg())
    loop.arm()
    for k in range(5):
        clock[0] += 0.01
        loop.keys.set(1.0, 0.0)
        loop.ball = BallState(110.0, 0.0, 200.0, 0.0, valid=True, t_s=clock[0])
        loop.tick(clock[0], 0.01)
        if k == 0:
            assert loop.snapshot()["guard"] == {"w": [0.0, 0.0], "side": [0.0, 0.0]}
    guard = loop.snapshot()["guard"]
    assert guard["w"][0] > 0.5 and guard["side"] == [1.0, 0.0]
    loop.mode = "balance"
    assert loop.snapshot()["guard"] is None
