import gc
import threading
import weakref
from concurrent.futures import wait

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
# the set place with its +x edge 30 mm out, past the edge tracker's reach
WIDE = configured(HALF, VIS) + [(30, 0), (0, 0), (0, 0), (30, 0)]


def board(dx=0.0, dy=0.0, deg=0.0) -> np.ndarray:
    """The board moved dx, dy px and turned deg from the set place, the ball at (30, 30) mm."""
    img = paint(DARK.copy(), at(SQUARE * HALF, dx, dy, deg), 210)
    x, y = at([(30.0, 30.0)], dx, dy, deg)[0]
    return ball(img, round(x), round(y), R_PX)


@pytest.fixture
def hold():
    """hold(pipe): its searches wait until the gate it returns opens, at the latest at the end."""
    gate = threading.Event()

    def held(pipe):
        find = pipe.finder.find
        pipe.finder.find = lambda *args: gate.wait() and find(*args)
        return gate

    yield held
    gate.set()


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
    # a tipped board can let such a lock through the shape check; a looser one stands in here
    stale = pipe.plate = PlateTracker({**VIS, "plate_shape_tol": 0.2}, HALF, WIDE)
    img = ball(table(), 700, 340, R_PX)
    o = pipe.step(img, 0.0)
    assert not o.occluded and stale.seen == 3 and abs(o.ball.x_mm - 30) > 10
    assert pipe.search.result()[0].seen == 4
    # the next frame hides that edge again: the find still counts the four it saw
    cv2.rectangle(img, (round(640 + 2 * HALF), 150), (1279, 650), 210, -1)
    o = pipe.step(img, DT)
    assert pipe.plate is not stale and pipe.plate.seen == stale.seen == 3
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx((30, 30), abs=0.5)


def test_a_lost_tracker_is_replaced_even_when_it_saw_as_many_edges():
    pipe = Pipeline(TRACKED)
    # a shape check that refuses everything: all four edges show and it is still lost
    lost = pipe.plate = PlateTracker({**VIS, "plate_shape_tol": 0.0}, HALF, configured(HALF, VIS))
    img = ball(table(), 700, 340, R_PX)
    assert pipe.step(img, 0.0).occluded and lost.seen == 4
    pipe.search.result()
    assert not pipe.step(img, DT).occluded and pipe.plate is not lost


def test_a_tracker_that_sees_all_four_edges_again_is_kept():
    pipe = Pipeline(TRACKED)
    stale = pipe.plate = PlateTracker({**VIS, "plate_shape_tol": 0.2}, HALF, WIDE)
    img = ball(table(), 700, 340, R_PX)
    pipe.step(img, 0.0)
    assert pipe.search.result()[0].seen == 4
    # by the time the find is in, a dark step shows where the tracker keeps its fourth edge
    cv2.rectangle(img, (round(640 + 2 * (HALF + 30)), 100), (1279, 700), 60, -1)
    pipe.step(img, DT)
    assert stale.seen == 4 and pipe.plate is stale


def hide(img: np.ndarray) -> np.ndarray:
    """A sheet as bright as the board against the +x edge of board(-120, -80)."""
    out = img.copy()
    cv2.rectangle(out, (round(at([(HALF, 0.0)], -120)[0, 0]), 0), (1200, 799), 210, -1)
    return out


def test_an_edge_hidden_in_every_frame_is_never_guessed():
    pipe = Pipeline(TRACKED)
    before, img, searches = pipe.plate, hide(board(-120, -80)), 0
    # 10 s apart, so every other step is past the rest and starts a search
    for t in range(0, 60, 10):
        assert pipe.step(img, t).occluded and pipe.plate is before
        if pipe.search is not None:
            searches += 1
            pipe.search.result()
    assert searches == 3


def test_a_search_from_a_clear_frame_locks_though_the_next_hides_an_edge_again():
    pipe = Pipeline(TRACKED)
    clear = board(-120, -80)
    pipe.step(hide(clear), 0.0)
    _, took = pipe.search.result()
    # nothing from the frame with the edge hidden
    assert pipe.step(clear, DT).occluded
    t = DT + 3 * took
    pipe.step(clear, t)
    pipe.search.result()
    o = pipe.step(hide(clear), t + DT)
    assert not o.occluded and pipe.plate.seen == 3
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx((30, 30), abs=0.5)


def test_a_find_the_board_has_moved_away_from_is_not_taken():
    pipe = Pipeline(TRACKED)
    pipe.step(board(120, -40, 15), 0.0)
    assert pipe.search.result()[0] is not None
    before = pipe.plate
    assert pipe.step(board(-120, 40, 15), DT).occluded and pipe.plate is before


def test_a_steady_four_edge_lock_never_starts_a_search():
    pipe = Pipeline(TRACKED)
    rng = np.random.default_rng(5)
    for i, (dx, dy) in enumerate(rng.uniform(-2, 2, (30, 2))):
        o = pipe.step(ball(table(dx, dy), 700, 340, R_PX), i * DT)
        assert not o.occluded and pipe.plate.seen == 4 and pipe.search is None


def test_one_search_at_a_time_then_a_rest_three_times_as_long(hold):
    pipe = Pipeline(TRACKED)
    gate = hold(pipe)
    pipe.step(DARK, 0.0)
    first = pipe.search
    for i in range(1, 20):
        pipe.step(DARK, i * DT)
        assert pipe.search is first
    gate.set()
    found, took = first.result()
    assert found is None
    pipe.step(DARK, 1.0)
    assert pipe.search is None
    pipe.step(DARK, 1.0 + 3 * took * 0.999)
    assert pipe.search is None
    pipe.step(DARK, 1.0 + 3 * took)
    assert pipe.search not in (None, first)
    pipe.search.result()


def test_a_search_that_ends_after_the_pipeline_is_gone_does_no_harm(hold):
    pipe = Pipeline(TRACKED)
    gate = hold(pipe)
    pipe.step(DARK, 0.0)
    search, gone = pipe.search, weakref.ref(pipe)
    del pipe
    gc.collect()
    # the search holds no pipeline: it only hands back what it found
    assert gone() is None
    gate.set()
    assert search.result()[0] is None


def test_a_search_that_fails_does_not_stop_the_frames():
    pipe = Pipeline(TRACKED)

    def broken(*args):
        raise np.linalg.LinAlgError("singular")

    pipe.finder.find = broken
    pipe.step(DARK, 0.0)
    wait([pipe.search])
    with pytest.raises(np.linalg.LinAlgError):
        pipe.step(DARK, DT)
    assert pipe.step(DARK, 2 * DT).occluded


@pytest.mark.parametrize("off_mm", [0.0, 5.0])
def test_a_three_edge_tracker_is_left_only_when_the_find_agrees_with_it(off_mm):
    pipe = Pipeline(TRACKED)
    # its +x search is out of reach, so it keeps that edge where it had it, off_mm out
    tr = pipe.plate = PlateTracker(VIS, HALF, WIDE)
    tr.corners = configured(HALF, VIS) + [(off_mm, 0), (0, 0), (0, 0), (off_mm, 0)]
    o = pipe.step(ball(table(), 700, 340, R_PX), 0.0)
    assert not o.occluded and tr.seen == 3
    assert pipe.search.result()[0].seen == 4
    # the next frame hides that edge, so the tracker still sees three
    img = ball(table(), 704, 340, R_PX)
    cv2.rectangle(img, (round(640 + 2 * HALF), 150), (1279, 650), 210, -1)
    o = pipe.step(img, DT)
    assert tr.seen == 3 and (pipe.plate is tr) == (off_mm == 0)
    # a swap clears the speed, so an agreeing find must not swap
    assert (o.ball.vx_mm_s > 100) == (off_mm == 0)
