import functools
import math

import numpy as np
import path_rig as rig
import pytest

from src.contracts import PlateCommand
from src.modes import make_mode
from src.modes.path import lengths, project, walk

pytestmark = pytest.mark.sim

NEAR_MM = 40.0  # parts of the path further along it than this are another part
OFF_MM = 12.0


def arc(centre, r, a0, a1, step=10.0):
    n = max(round(abs(a1 - a0) * r / step), 1)
    turns = [a0 + (a1 - a0) * i / n for i in range(n + 1)]
    return [(centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)) for a in turns]


def double_spiral(gap=12.0, r_in=14.0, r_out=70.0, step=10.0):
    """In along one arm and out along the other, gap mm apart, through an S at the middle:
    every turn has a later or earlier part of the path beside it, as on the drawn line. Both
    ends run on straight out, clear of the other arm."""
    b, arm, a = gap / math.pi, [], r_out * math.pi / gap
    while a > r_in / b:
        arm.append((b * a * math.cos(a), b * a * math.sin(a)))
        a -= step / (b * a)
    a = r_in / b
    tip = (r_in * math.cos(a), r_in * math.sin(a))
    s = arc((tip[0] / 2, tip[1] / 2), r_in / 2, a, a + math.pi)
    s += arc((-tip[0] / 2, -tip[1] / 2), r_in / 2, a, a - math.pi)[1:]
    out = (r_out + 3 * gap) / math.hypot(*arm[0])
    arm = rig.line((arm[0][0] * out, arm[0][1] * out), arm[0])[:-1] + arm
    return arm + s + [(-x, -y) for x, y in arm[::-1]]


SPIRAL = double_spiral()


def near(path, cum, p, lo, hi):
    """Distance from p to the part of the path between lo and hi mm along it."""
    best = math.inf
    for i in range(len(path) - 1):
        a, b = max(lo, cum[i]), min(hi, cum[i + 1])
        if a <= b:
            t = min(max(project(p, path[i], path[i + 1]), a - cum[i]), b - cum[i])
            best = min(best, math.dist(p, walk(path, i, t, 0.0)))
    return best


def run(path, back=False, late_s=rig.SEEN_LATE_S, lean=(0.0, 0.0), gaps=rig.GAPS_S, each=None):
    """A run with the rig keys and where the mode puts the ball on the path at every tick;
    each, when given, is called with the cfg and the time before every tick."""
    cfg = rig.rig_cfg(path, rig.RIG_CONTROL)
    mode, places, track, tilts = make_mode("path", cfg), [], [], []

    def watched(ball, intent, cfg):
        if each:
            each(cfg, len(places) / cfg["control"]["sim_tick_hz"])
        cmd = mode(ball, intent, cfg)
        places.append(mode.open.s)
        tilts.append((cmd.tilt_x_deg, cmd.tilt_y_deg))
        return cmd

    start, end = (path[-1], path[0]) if back else (path[0], path[-1])
    kw = dict(gaps=gaps, lean=lean, cfg=cfg, mode=watched, track=track, late_s=late_s)
    r = rig.follow(path, start, end, 120.0, **kw)
    n = round(r["time_s"] * 100) if r["finished"] else len(track)
    return r, mode.open.path, places[:n], track[:n], tilts[:n]


def passed_off(path, places, track):
    """Share of the path the mode's place passed with the ball more than OFF_MM from it."""
    pairs = zip(places, places[1:], track[1:], strict=False)
    off = sum(b - a for a, b, p in pairs if math.dist(p, walk(path, 0, 0.0, b)) > OFF_MM)
    return off / lengths(path)[-1]


def stalled_s(places, dt=0.01):
    """Longest time the mode's place stood still."""
    longest, since = 0.0, 0
    for k in range(1, len(places)):
        since = since if places[k] == places[since] else k
        longest = max(longest, (k - since) * dt)
    return longest


def missed(path, track, mm):
    """Share of the path, in 2 mm steps, that the true ball never came within mm of."""
    line = np.array([walk(path, 0, 0.0, d) for d in np.arange(0.0, lengths(path)[-1], 2.0)])
    t = np.array(track)
    near_mm = np.full(len(line), np.inf)
    for k in range(0, len(t), 2000):
        d = np.hypot(*(line[:, None, i] - t[None, k : k + 2000, i] for i in (0, 1)))
        near_mm = np.minimum(near_mm, d.min(axis=1))
    return float((near_mm > mm).mean())


@functools.cache
def judged(back=False, late_s=rig.SEEN_LATE_S, lean=(0.0, 0.0), gaps=rig.GAPS_S):
    """A run on the double spiral judged at the place the mode itself gives the ball: the
    share of the path passed more than OFF_MM off it, and the share of 20 Hz rows with the
    ball nearer another part of the path than its own."""
    r, path, places, track, _ = run(SPIRAL, back, late_s, lean, gaps)
    cum = lengths(path)
    rows = [(p, s) for p, s in zip(track[::5], places[::5], strict=True)]
    other = sum(
        min(near(path, cum, p, -1, s - NEAR_MM), near(path, cum, p, s + NEAR_MM, math.inf))
        < near(path, cum, p, s - NEAR_MM, s + NEAR_MM)
        for p, s in rows
    )
    off = passed_off(path, places, track)
    return r, off, other / len(rows), missed(path, track, 5.0), missed(path, track, 8.0)


def test_the_double_spiral_has_another_part_of_itself_within_13_mm_almost_everywhere():
    cum = lengths(SPIRAL)
    beside = [
        near(SPIRAL, cum, p, -1, s - NEAR_MM) < 13
        or near(SPIRAL, cum, p, s + NEAR_MM, math.inf) < 13
        for p, s in zip(SPIRAL, cum, strict=True)
    ]
    assert sum(beside) / len(beside) > 0.75 and cum[-1] > 1300


def test_the_rig_can_see_the_ball_later():
    seen = []

    def spy(ball, intent, cfg):
        seen.append(ball)
        return PlateCommand(0.0, 0.0)

    rig.follow(rig.STRAIGHT, seconds=3.0, mode=spy, late_s=0.18)
    servo_s = rig.rig_cfg(rig.STRAIGHT)["control"]["servo_dead_time_s"]
    assert 3.0 - seen[-1].t_s == pytest.approx(0.18 - servo_s, abs=0.011)


# measured: 0.4% passed off, 1.5 to 1.6% of rows nearer another part, the true ball more than 5 mm
# from 3.0 to 3.1% of the line (today's code 13%, 77 to 78%, 41%; aimed only along the line, without
# the turn its speed needs, 5.1%), 40 s, resting 4.2 to 4.6 mm from the end
@pytest.mark.parametrize("back", [False, True], ids=["in first", "out first"])
def test_the_double_spiral_is_followed_on_its_own_turn_to_the_end_both_ways(back):
    r, off, other, five, _ = judged(back)
    assert off < 0.05 and other < 0.03 and five < 0.04
    assert r["finished"] and r["time_s"] < 75.0 and r["end_mm"] < 10.0 and r["end_mm_s"] < 1.0


# measured with the ball seen as late as on the rig: 1.0 to 1.1% passed off, more than 8 mm from 4.2
# to 5.2% of the line (today's code 48 to 50%, 21%)
@pytest.mark.parametrize("back", [False, True], ids=["in first", "out first"])
def test_seen_as_late_as_on_the_rig_the_double_spiral_is_still_kept_to(back):
    r, off, _, _, eight = judged(back, late_s=0.18)
    assert off < 0.05 and eight < 0.08 and r["finished"] and r["time_s"] < 75.0


# measured: 0.0 to 0.1% passed off, more than 8 mm from 1.8 to 4.5% of the line (today's code 30 to
# 42%, 17 to 21%), 41 to 44 s
@pytest.mark.parametrize(
    "back, lean",
    [(False, (0.5, 0.0)), (False, (-0.5, 0.0)), (True, (0.0, 0.5)), (True, (0.0, -0.5))],
    ids=["+x in first", "-x in first", "+y out first", "-y out first"],
)
def test_a_board_leaning_half_a_degree_neither_stalls_the_ball_nor_pushes_it_off(back, lean):
    r, off, _, _, eight = judged(back, lean=lean)
    assert off < 0.05 and eight < 0.06 and r["finished"] and r["time_s"] < 75.0


# measured: 6.7 to 6.8 s, the place never still for more than 1.0 s
@pytest.mark.parametrize("lean", [(0.0, 0.5), (0.0, -0.5)], ids=["+y", "-y"])
def test_a_lean_across_the_line_holding_the_ball_off_it_does_not_stall_the_run(lean):
    straight = rig.line((-100.0, 0.0), (100.0, 0.0))
    r, path, places, track, _ = run(straight, lean=lean, gaps=())
    assert r["finished"] and r["time_s"] < 15.0 and stalled_s(places) < 3.0
    assert passed_off(path, places, track) < 0.05


def test_the_bend_speed_can_change_mid_run_without_a_jump_or_losing_the_place():
    def speed(cfg, t):
        cfg["control"]["path_bend_mm_s"] = 60.0 if 12.0 <= t < 24.0 else 15.0

    r, path, places, track, tilts = run(SPIRAL, each=speed)
    steps = [math.dist(a, b) for a, b in zip(tilts, tilts[1:], strict=False)]
    # measured: 38 s, 0.0% passed off, 0.30 deg at the changes against 1.86 deg at most otherwise
    assert r["finished"] and r["time_s"] < 75.0 and passed_off(path, places, track) < 0.05
    assert all(b >= a for a, b in zip(places, places[1:], strict=False))
    assert max(steps[1195:1205] + steps[2395:2405]) < 0.5 * max(steps)


def test_seen_as_late_as_on_the_rig_gentle_lines_are_kept_to_at_least_as_well_as_before():
    line, track = rig.line((-100.0, 0.0), (100.0, 0.0)), []
    cfg = rig.rig_cfg(line, rig.RIG_CONTROL)
    rig.follow(line, (-100.0, 20.0), seconds=5.0, gaps=(), cfg=cfg, track=track, late_s=0.18)
    wide = arc((-80.0, 0.0), 80.0, -math.pi / 2, 0.0) + arc((80.0, 0.0), 80.0, math.pi, math.pi / 2)
    _, path, _, s_track, _ = run(wide, late_s=0.18, gaps=())
    # measured: started 20 mm beside a straight, swinging at most 10.3 mm from 3 to 5 s (today's
    # code 9.6, held back on the straight too 12.3); on an S of 80 mm bends 8.7% of the line missed
    # by more than 8 mm (turning on the whole speed, not the speed along it 11.9%, today 12.7%)
    assert max(abs(y) for _, y in track[300:500]) < 11.5 and missed(path, s_track, 8.0) < 0.10
