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
    STILL,
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
LEVEL = np.array([])


def snap(t_s: float, mm=(0.0, 0.0), px=A, v: float = 0.0, valid: bool = True, r=R) -> dict:
    ball = {"x_mm": mm[0], "y_mm": mm[1], "vx_mm_s": v, "vy_mm_s": 0.0, "valid": valid}
    return {"t_s": t_s, "ball": ball, "ball_px": [*px, r] if px else None}


def shot(t_s: float = 100.0, stretch: int = 0, b_px=None, **ball) -> rec.Shot:
    """A picture between two snapshots 50 ms apart, the ball at b_px by the second one."""
    after = snap(t_s + 0.05, **{**ball, "px": b_px or ball.get("px", A)})
    return rec.Shot(Path(f"{t_s:010.3f}.jpg"), snap(t_s, **ball), after, stretch)


def lying(start: float = 90.0, end: float = 110.0) -> Track:
    t = np.arange(start, end, 0.01)
    return Track(t, np.zeros((len(t), 2)))


def mark(shots, found, arrived=(), second=None, tilted=LEVEL) -> list:
    return labels(shots, found, set(arrived), tilted, BENCH, second)


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


def test_a_moving_balls_label_is_the_path_between_snapshots():
    moving = shot(b_px=B, v=800.0)
    # 15 px off the middle of the path, 25 px from either end
    assert mark([moving], [], (), at(120.0, 115.0)) == [Label(A, B, R, SECOND, False, True)]
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
    # 1 mm/s: 3.05 mm over the three seconds before the picture and its 50 ms
    drift = lying()
    drift.xy[:, 0] = np.linspace(0.0, 20.0, len(drift.t))
    assert not rest(shot(), drift)
    # seen for two and a half seconds only
    assert not rest(shot(), lying(start=97.5))


def test_a_still_ball_between_agreeing_sightings_is_labelled_while_unseen():
    # seen until 101.5, lost for two seconds, then seen again; a picture every second
    seen = [lying(90.0, 101.5), lying(103.5, 110.0)]

    def gap(found=seen, arrived=(0, 1), tilted=LEVEL, last=((1.0, 0.0), 0.0), stretch=0, **before):
        shots = [
            shot(100.0),
            shot(101.0, **before),
            shot(102.0, **GONE),
            shot(103.0, **GONE),
            shot(104.0, stretch, mm=last[0], v=last[1], px=(101.0, 100.0)),
            shot(105.0, stretch),
        ]
        return mark(shots, found, arrived, tilted=tilted)[2:4]

    # at the place it was last seen
    still = [Label(A, A, R, STILL, True, None)] * 2
    lost = [UNSEEN] * 2
    assert gap() == still
    # afterwards it lies somewhere else
    assert gap(last=((3.0, 0.0), 0.0)) == lost
    # it was rolling when last seen, or when found again
    assert gap(v=5.0) == lost
    assert gap(last=((1.0, 0.0), 5.0)) == lost
    # the sighting before has nothing behind it, or the one after
    assert gap(arrived=(1,)) == lost
    assert gap(arrived=(0,)) == lost
    # the plate was asked to tilt while the ball was out of sight; before or after is no matter
    assert gap(tilted=np.array([102.5])) == lost
    assert gap(tilted=np.array([100.5, 104.5])) == still
    # found again, but tracked for two and a half seconds only
    assert gap(found=[seen[0], lying(103.5, 106.0)]) == lost
    # the recording itself has a gap in between
    assert gap(stretch=1) == lost
