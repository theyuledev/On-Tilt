import functools
import math

import path_rig as rig
import pytest

from src.contracts import PlateCommand
from src.modes import make_mode
from src.modes.path import walk

pytestmark = pytest.mark.sim

FAR = (1e6, 1e6)  # a loop has no end to finish at
OFF_MM = 12.0


def closed(points):
    return [*points, points[0]]


def circle(r, centre=(0.0, 0.0), n=None, a0=0.0):
    n = n or max(round(2 * math.pi * r / 10.0), 8)
    turns = [a0 + 2 * math.pi * i / n for i in range(n)]
    return [(centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)) for a in turns]


def heart(n=60, k=5.0):
    """About 160 mm wide, with its sharp point at the bottom and the dip at the top."""
    out = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        out.append((k * x, k * y + 15.0))
    return out


def eight(n=72, a=100.0, b=60.0):
    """A figure eight crossing itself at the middle."""
    return [
        (a * math.sin(t), b * math.sin(t) * math.cos(t))
        for t in (2 * math.pi * i / n for i in range(n))
    ]


TAIL = rig.line((-110.0, -80.0), (-50.0, 0.0))
LOOPS = {
    "circle r 80": closed(circle(80.0)),
    "heart": closed(heart()),
    "figure eight": closed(eight()),
    # the loop starts and ends exactly on the tail's last point
    "tail into loop": TAIL + circle(50.0, a0=math.pi)[1:] + [TAIL[-1]],
    # past the reach on the right, pulled in to run along it
    "by the edge": closed(
        rig.line((40.0, -60.0), (138.0, -60.0)) + rig.line((138.0, 60.0), (40.0, 60.0))
    ),
}


@functools.cache
def lapped(name, late_s=rig.SEEN_LATE_S, gap=None, seconds=90.0):
    """A run on a loop with the rig keys: laps, how far along the mode took the ball in all at
    every tick, the share of that passed with the true ball more than OFF_MM from its place,
    the longest the place stood still and the room left to the rim. gap, (from, to) s, is a
    stretch without a heartbeat: the loop calls resume() and levels the plate."""
    path = LOOPS[name]
    cfg = rig.rig_cfg(path, rig.RIG_CONTROL)
    mode, track, places, hz = make_mode("path", cfg), [], [], cfg["control"]["sim_tick_hz"]

    def watched(ball, intent, cfg):
        stale = gap and gap[0] <= len(places) / hz < gap[1]
        if stale:
            mode.resume()
        cmd = mode(ball, intent, cfg)
        gone = mode.laps * getattr(mode, "lap", 0.0) + mode.s
        places.append((gone, walk(mode.path, 0, 0.0, mode.s)))
        return PlateCommand(0.0, 0.0) if stale else cmd

    r = rig.follow(path, None, FAR, seconds, cfg=cfg, mode=watched, track=track, late_s=late_s)
    gone = [g for g, _ in places]
    pairs = zip(places, places[1:], track[1:], strict=False)
    off = sum(b[0] - a[0] for a, b, at in pairs if math.dist(at, b[1]) > OFF_MM)
    longest, since = 0, 0
    for k in range(1, len(gone)):
        since = since if gone[k] == gone[since] else k
        longest = max(longest, k - since)
    rim = (cfg["rig"]["plate_inner_mm"] - cfg["rig"]["ball_d_mm"]) / 2
    room = rim - r["far_mm"] - cfg["control"]["guard_margin_mm"]
    return {"laps": mode.laps, "gone": gone, "off": off / gone[-1], "still_s": longest / hz}, room


def test_the_tail_runs_into_the_loop_exactly_on_its_last_point():
    path = LOOPS["tail into loop"]
    assert path[-1] == TAIL[-1] and path.index(path[-1]) == len(TAIL) - 1


# measured in 90 s: 7, 5, 6, 10 and 7 laps, 0.0% passed off, the place still for 0.9 s at most (the
# longer blind gap), 0.8 mm to spare by the edge
@pytest.mark.parametrize("name", LOOPS)
def test_every_loop_is_lapped_on_its_line_and_never_stops(name):
    r, room = lapped(name)
    assert r["laps"] >= 3 and r["off"] < 0.05 and r["still_s"] < 3.0 and room > 0
    # the place only ever goes on, round the seam too
    assert all(b >= a for a, b in zip(r["gone"], r["gone"][1:], strict=False))


# measured: 848 mm along before the gap, 878 after it, 7 laps as without it
def test_a_gap_in_the_heartbeat_mid_lap_goes_on_where_it_was():
    r, _ = lapped("circle r 80", gap=(20.0, 21.0))
    plain, _ = lapped("circle r 80")
    assert r["gone"][2101] >= r["gone"][1999] > 503.0 and r["laps"] >= plain["laps"] - 1
    assert all(b >= a for a, b in zip(r["gone"], r["gone"][1:], strict=False))


# measured: 7 laps, 2.2% passed off
def test_seen_as_late_as_on_the_rig_a_loop_is_still_lapped():
    r, room = lapped("circle r 80", late_s=0.18)
    assert r["laps"] >= 3 and r["off"] < 0.10 and r["still_s"] < 3.0 and room > 0
