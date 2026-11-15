import gc
import threading
import weakref

import cv2
import numpy as np
import pytest

from src.vision.pipeline import Pipeline
from src.vision.plate_find import SQUARE, PlateTracker, configured
from tests.unit.test_balls import ball
from tests.unit.test_plate_find import at, paint
from tests.vision.test_pipeline_plate import DT, HALF, R_PX, TRACKED, VIS, table

pytestmark = pytest.mark.unit

DARK = np.full((800, 1280), 60, np.uint8)


def board(dx=0.0, dy=0.0, deg=0.0) -> np.ndarray:
    """The board alone, moved dx, dy px and turned deg from the set place, the ball on it at
    (30, 30) mm."""
    img = paint(DARK.copy(), at(SQUARE * HALF, dx, dy, deg), 210)
    x, y = at([(30.0, 30.0)], dx, dy, deg)[0]
    return ball(img, round(x), round(y), R_PX)


@pytest.fixture
def gate():
    g = threading.Event()
    yield g
    g.set()


def held(pipe, gate):
    """Every search waits for the gate before it looks."""
    find = pipe.finder.find

    def wait_then_find(*args):
        gate.wait()
        return find(*args)

    pipe.finder.find = wait_then_find


def test_a_board_far_from_its_set_place_is_found_beside_the_grab_thread():
    pipe = Pipeline(TRACKED)
    img = board(120, -40, 15)
    assert pipe.step(img, 0.0).occluded and pipe.search is not None
    pipe.search.result()
    o = pipe.step(img, DT)
    assert not o.occluded and pipe.plate.seen == 4
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx((30, 30), abs=0.5)
    # the map jumped onto the board, which is no speed of the ball's
    assert (o.ball.vx_mm_s, o.ball.vy_mm_s) == (0.0, 0.0)


def test_a_lock_with_an_edge_out_of_reach_is_replaced_by_a_four_edge_find():
    pipe = Pipeline(TRACKED)
    # the +x edge 30 mm out, past the edge tracker's reach. A tipped board can let such a lock
    # through the shape check; a looser check stands in for that here
    wide = configured(HALF, VIS) + [(30, 0), (0, 0), (0, 0), (30, 0)]
    stale = pipe.plate = PlateTracker({**VIS, "plate_shape_tol": 0.2}, HALF, wide)
    img = ball(table(), 700, 340, R_PX)
    o = pipe.step(img, 0.0)
    assert not o.occluded and stale.seen == 3 and abs(o.ball.x_mm - 30) > 10
    pipe.search.result()
    o = pipe.step(img, DT)
    assert pipe.plate is not stale and pipe.plate.seen == 4
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx((30, 30), abs=0.5)


def test_a_lost_tracker_is_replaced_even_when_it_saw_as_many_edges():
    pipe = Pipeline(TRACKED)
    # a shape check that refuses everything: all four edges show and it is still lost
    lost = pipe.plate = PlateTracker({**VIS, "plate_shape_tol": 0.0}, HALF, configured(HALF, VIS))
    img = ball(table(), 700, 340, R_PX)
    assert pipe.step(img, 0.0).occluded and lost.seen == 4
    pipe.search.result()
    assert not pipe.step(img, DT).occluded and pipe.plate is not lost


def test_a_find_that_sees_no_more_edges_leaves_the_tracker():
    img = DARK.copy()
    # a sheet as bright as the board against its +x side hides that edge from both
    cv2.rectangle(img, (900, 150), (1279, 650), 210, -1)
    img = ball(paint(img, at(SQUARE * HALF), 210), 700, 340, R_PX)
    pipe = Pipeline(TRACKED)
    o = pipe.step(img, 0.0)
    before = pipe.plate
    assert not o.occluded and before.seen == 3
    assert pipe.search.result().seen == 3
    pipe.step(img, DT)
    assert pipe.plate is before


def test_a_find_the_board_has_moved_away_from_is_not_taken():
    pipe = Pipeline(TRACKED)
    pipe.step(board(120, -40, 15), 0.0)
    assert pipe.search.result() is not None
    before = pipe.plate
    assert pipe.step(board(-120, 40, 15), DT).occluded and pipe.plate is before


def test_a_steady_four_edge_lock_never_starts_a_search():
    pipe = Pipeline(TRACKED)
    rng = np.random.default_rng(5)
    for i, (dx, dy) in enumerate(rng.uniform(-2, 2, (30, 2))):
        o = pipe.step(ball(table(dx, dy), 700, 340, R_PX), i * DT)
        assert not o.occluded and pipe.plate.seen == 4 and pipe.search is None


def test_one_search_at_a_time_then_a_rest_three_times_as_long(gate):
    pipe = Pipeline(TRACKED)
    held(pipe, gate)
    pipe.step(DARK, 0.0)
    first = pipe.search
    for i in range(1, 20):
        pipe.step(DARK, i * DT)
        assert pipe.search is first
    gate.set()
    assert first.result() is None
    # taken up 1 s after it started, so the next one waits until 4 s
    pipe.step(DARK, 1.0)
    assert pipe.search is None
    pipe.step(DARK, 3.99)
    assert pipe.search is None
    pipe.step(DARK, 4.0)
    assert pipe.search not in (None, first)
    pipe.search.result()


def test_a_search_that_ends_after_the_pipeline_is_gone_does_no_harm(gate):
    pipe = Pipeline(TRACKED)
    held(pipe, gate)
    pipe.step(DARK, 0.0)
    search, gone = pipe.search, weakref.ref(pipe)
    del pipe
    gc.collect()
    # the search holds no pipeline: it only hands back what it found
    assert gone() is None
    gate.set()
    assert search.result() is None
