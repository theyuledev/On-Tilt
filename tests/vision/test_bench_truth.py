from pathlib import Path

import numpy as np
import pytest

from src.vision.bench import rec
from src.vision.bench.track import Track
from src.vision.bench.truth import (
    ALONE,
    ARRIVAL,
    DISPUTED,
    SECOND,
    UNREADABLE,
    UNSEEN,
    Label,
    labels,
)

pytestmark = pytest.mark.unit

BENCH = {"rest_mm_s": 5.0, "rest_mm": 3.0, "rest_s": 3.0}
A, B = (100.0, 100.0), (140.0, 100.0)
R = 20.0
GONE = {"valid": False, "px": None}


def snap(t_s: float, mm=(0.0, 0.0), px=A, v: float = 0.0, valid: bool = True, r=R) -> dict:
    ball = {"x_mm": mm[0], "y_mm": mm[1], "vx_mm_s": v, "vy_mm_s": 0.0, "valid": valid}
    return {"t_s": t_s, "ball": ball, "ball_px": [*px, r] if px else None}


def shot(t_s: float = 100.0, b_px=None, **ball) -> rec.Shot:
    """A picture between two snapshots 50 ms apart, the ball at b_px by the second one."""
    after = snap(t_s + 0.05, **{**ball, "px": b_px or ball.get("px", A)})
    return rec.Shot(Path(f"{t_s:010.3f}.jpg"), snap(t_s, **ball), after, 0)


def lying(start: float = 90.0, end: float = 110.0) -> Track:
    t = np.arange(start, end, 0.01)
    return Track(t, np.zeros((len(t), 2)))


def mark(shots, found, arrived=(), second=None) -> list:
    return labels(shots, found, set(arrived), BENCH, second)


def at(x: float, y: float, r: float = R):
    """A second detector that finds a ball of radius r there."""
    return lambda _shot: (x, y, r)


def test_a_frame_on_a_track_that_arrived_is_labelled():
    assert mark([shot(b_px=B)], [lying()], {0}) == [Label(A, B, R, ARRIVAL, True, None)]


def test_a_frame_without_support_gets_no_label_and_a_reason():
    assert mark([shot()], [lying()]) == [ALONE]
    # the two snapshots sit on different tracks: nothing carries the arrival across
    cut = [lying(90.0, 100.02), lying(100.03, 110.0)]
    assert mark([shot()], cut, {0, 1}) == [ALONE]
    assert mark([shot(**GONE)], [lying()], {0}) == [UNSEEN]
    assert mark([shot(px=None)], [lying()], {0}) == [UNSEEN]
    # seen at only one of the two snapshots around the picture
    for gone in ({"valid": False}, {"px": None}):
        first = rec.Shot(Path("x.jpg"), snap(100.0, **gone), snap(100.05), 0)
        second = rec.Shot(Path("x.jpg"), snap(100.0), snap(100.05, **gone), 0)
        assert mark([first, second], [lying()], {0}) == [UNSEEN, UNSEEN]


def test_a_moving_balls_label_is_the_path_between_snapshots():
    moving = shot(b_px=B, v=800.0)
    # 15 px off the middle of the path, 25 px from either end
    assert mark([moving], [], (), at(120.0, 115.0)) == [Label(A, B, R, SECOND, False, True)]
    assert mark([moving], [], (), at(120.0, 120.0)) == [Label(A, B, R, SECOND, False, True)]
    assert mark([moving], [], (), at(120.0, 121.0)) == [DISPUTED]
    assert mark([moving], [], (), at(161.0, 100.0)) == [DISPUTED]
    assert mark([moving], [], (), at(79.0, 100.0)) == [DISPUTED]


def test_the_second_detector_judges_by_its_own_ball_size():
    # the live tracker recorded a blob three times the ball
    big = shot(r=60.0)
    assert mark([big], [], (), at(130.0, 100.0)) == [DISPUTED]
    assert mark([big], [], (), at(115.0, 100.0)) == [Label(A, A, 60.0, SECOND, False, True)]


def test_a_disagreement_is_disputed_not_wrong():
    far = at(300.0, 300.0)
    # with an arrival behind it the label stands and the dispute is written down
    assert mark([shot()], [lying()], {0}, far) == [Label(A, A, R, ARRIVAL, True, False)]
    # with nothing else behind it there is no label, and no verdict either
    assert mark([shot()], [lying()], (), far) == [DISPUTED]


def test_an_unreadable_picture_is_skipped_with_its_reason():
    assert mark([shot()], [lying()], {0}, lambda s: None) == [UNREADABLE]


def test_at_rest_needs_a_slow_ball_that_stayed_put():
    def rest(one: rec.Shot, track: Track) -> bool:
        return mark([one], [track], {0})[0].at_rest

    assert rest(shot(), lying())
    assert not rest(shot(v=5.0), lying())
    # fast at just one of the two snapshots around the picture
    assert not rest(rec.Shot(Path("x.jpg"), snap(100.0), snap(100.05, v=5.0), 0), lying())
    assert not rest(rec.Shot(Path("x.jpg"), snap(100.0, v=5.0), snap(100.05), 0), lying())
    # 0.99 mm/s: 2.97 mm over the three seconds before the picture, 3.02 mm with its 50 ms
    drift = lying()
    drift.xy[:, 0] = 0.99 * (drift.t - drift.t[0])
    assert not rest(shot(), drift)
    # exactly the limit away a second before
    stepped = lying()
    stepped.xy[stepped.t < 99.0, 0] = BENCH["rest_mm"]
    assert not rest(shot(), stepped)
    # it rolls off after the picture
    later = lying()
    later.xy[:, 0] = np.clip(50.0 * (later.t - 102.0), 0.0, None)
    assert rest(shot(), later)
    # seen for two and a half seconds only
    assert not rest(shot(), lying(start=97.5))
    # anywhere on the plate
    away = lying()
    away.xy[:] = (50.0, 30.0)
    assert rest(shot(), away)
    # 2 mm one way, then 2 mm the other: 4 mm from where it ends up
    swung = lying()
    swung.xy[:, 0] = np.select([swung.t < 98.0, swung.t < 99.0], [0.0, -2.0], 2.0)
    assert not rest(shot(), swung)
