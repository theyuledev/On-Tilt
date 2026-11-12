from collections import Counter

import pytest

from src.vision.bench import track
from src.vision.bench.track import arrivals, holding, tracks

pytestmark = pytest.mark.unit

TOP = 500.0  # mm/s
HOLE = 0.04
AT = 24.0
TICK = 0.01


def row(t_s: float, x: float, valid: bool = True, frame_t_s: float | None = None) -> dict:
    ball = {"x_mm": x, "y_mm": 0.0, "valid": valid, "t_s": t_s if frame_t_s is None else frame_t_s}
    return {"t_s": t_s, "ball": ball}


def rolling(xs: list[float], t0: float = 100.0) -> list[dict]:
    return [row(t0 + i * TICK, x) for i, x in enumerate(xs)]


def trial(target=(0.0, 0.0), start=100.0, reached=1.0, **more) -> dict:
    more = {"ball_lost": False, "stuck": False, **more}
    return {"target": list(target), "t_start": start, "reached_s": reached, **more}


def to_the_target() -> list[dict]:
    """From 60 mm out to the target in one second, then a second of rest there."""
    return rolling([-60.0 + 0.6 * i for i in range(100)] + [0.0] * 100)


def test_a_step_above_top_speed_breaks_the_chain():
    # 4 mm a tick is 400 mm/s, 6 mm a tick is 600
    got = tracks(rolling([0.0, 4.0, 8.0, 14.0, 18.0]), TOP, HOLE)
    assert [list(tr.xy[:, 0]) for tr in got] == [[0.0, 4.0, 8.0], [14.0, 18.0]]
    assert list(got[1].t) == pytest.approx([100.03, 100.04])


def test_a_lost_stretch_breaks_the_chain():
    rows = rolling([0.0, 1.0, 2.0, 3.0, 4.0])
    rows[2]["ball"]["valid"] = False
    got = tracks(rows, TOP, HOLE)
    assert [list(tr.xy[:, 0]) for tr in got] == [[0.0, 1.0], [3.0, 4.0]]


def test_a_hole_longer_than_the_tracker_coasts_breaks_the_chain():
    # the last row's loop clock steps back while its camera frame is the next one
    back = row(99.0, 0.0, frame_t_s=100.1)
    rows = [row(100.0, 0.0), row(100.03, 0.0), row(100.09, 0.0), back]
    assert [len(tr.t) for tr in tracks(rows, TOP, HOLE)] == [2, 1, 1]


def test_rows_of_one_camera_frame_stay_joined_unless_the_ball_moved():
    same = [row(100.0, 5.0, frame_t_s=99.99), row(100.01, 5.0, frame_t_s=99.99)]
    assert len(tracks(same, TOP, HOLE)) == 1
    moved = [row(100.0, 5.0, frame_t_s=99.99), row(100.01, 5.5, frame_t_s=99.99)]
    assert len(tracks(moved, TOP, HOLE)) == 2


def test_a_hole_in_the_camera_clock_breaks_the_chain_too():
    # the loop wrote its rows on time while the camera stalled
    def stalled(for_s: float) -> list[dict]:
        return [row(100.0, 5.0, frame_t_s=0.0), row(100.01, 5.0, frame_t_s=for_s)]

    assert len(tracks(stalled(HOLE), TOP, HOLE)) == 1
    assert len(tracks(stalled(HOLE + 0.001), TOP, HOLE)) == 2
    # a hole of exactly the coast time on both clocks is still one track
    assert len(tracks([row(0.0, 5.0), row(HOLE, 5.0)], TOP, HOLE)) == 1


def test_holding_wants_one_track_over_the_whole_span():
    got = tracks(rolling([0.0, 1.0, 2.0, 9.0, 10.0]), TOP, HOLE)
    a, b = got
    assert holding(got, a.t[0], a.t[-1]) == 0 and holding(got, b.t[0], b.t[-1]) == 1
    assert holding(got, a.t[-1], b.t[0]) is None and holding(got, a.t[0] - TICK, a.t[-1]) is None


def test_arrival_at_a_driver_target_is_an_anchor():
    got = tracks(to_the_target(), TOP, HOLE)
    closing = {"ended": "time is up", "settings_restored": {}}
    assert arrivals([trial(), closing], got, AT) == (
        {0},
        Counter({track.ARRIVED: 1}),
        0,
    )


def test_a_mark_under_the_target_without_travel_is_no_anchor():
    still = tracks(rolling([5.0] * 200), TOP, HOLE)
    assert arrivals([trial()], still, AT) == (set(), Counter({track.NO_TRAVEL: 1}), 0)
    # the driver saw it there on its first tick
    assert arrivals([trial(reached=0.0)], still, AT).trials == Counter({track.NO_TRAVEL: 1})
    # right on the circle is still at the target
    on = tracks(rolling([-24.0 + 0.24 * i for i in range(100)] + [0.0] * 100), TOP, HOLE)
    assert arrivals([trial()], on, AT).trials == Counter({track.NO_TRAVEL: 1})
    # the same thing 60 mm from another target is not travel either: it never gets there
    assert arrivals([trial(target=(65.0, 0.0), reached=None)], still, AT).tracks == set()
    # just outside the circle at the start counts as coming from somewhere else
    edge = tracks(rolling([-24.5 + 0.245 * i for i in range(100)] + [0.0] * 100), TOP, HOLE)
    assert arrivals([trial()], edge, AT).tracks == {0}
    # what counts is where it is when the trial starts, not where its track began
    there = arrivals([trial(start=101.2, reached=0.1)], tracks(to_the_target(), TOP, HOLE), AT)
    assert there.trials == Counter({track.NO_TRAVEL: 1})


def test_a_trial_that_never_reached_its_target_gives_no_anchor():
    got = tracks(to_the_target(), TOP, HOLE)
    trials = [trial(reached=None), trial(ball_lost=True), trial(stuck=True)]
    assert arrivals(trials, got, AT) == (
        set(),
        Counter({track.NEVER: 1, track.LOST: 1, track.STUCK: 1}),
        0,
    )


def test_a_missing_stuck_field_is_unknown_and_counted():
    got = tracks(to_the_target(), TOP, HOLE)
    old = {k: v for k, v in trial().items() if k != "stuck"}
    assert arrivals([old, trial()], got, AT) == ({0}, Counter({track.ARRIVED: 2}), 1)


def test_an_arrival_after_a_break_on_the_way_is_no_anchor():
    rows = to_the_target()
    rows[50]["ball"]["valid"] = False
    got = tracks(rows, TOP, HOLE)
    assert arrivals([trial()], got, AT) == (set(), Counter({track.BROKEN: 1}), 0)
    # a trial that starts after the break rides the second track
    assert arrivals([trial(start=100.55, reached=0.45)], got, AT).tracks == {1}
