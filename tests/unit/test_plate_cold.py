import cv2
import numpy as np
import pytest

from src.vision.calib import px_to_plate_mm
from src.vision.plate_cold import ColdFinder, squares, top_maps
from src.vision.plate_find import SQUARE, configured
from tests.unit.test_plate_find import CALIB, HALF, TRUE, VIS, at, paint, scene

# a picture with no board must not fill the log with numpy warnings either
pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("error")]

SET = configured(HALF, VIS)


def truth(dx=0.0, dy=0.0, deg=0.0) -> np.ndarray:
    """The drawn board's corners in the calibration's mm, named like the set ones."""
    return px_to_plate_mm(at(TRUE, dx, dy, deg), CALIB)


def find(img):
    return ColdFinder(VIS, HALF).find(img, SET)


def candidates(img) -> list:
    x, y, ok, _ = top_maps(VIS, HALF, img.shape)
    top = cv2.remap(img, x, y, cv2.INTER_LINEAR).astype(np.float32) * ok
    return squares(top, ok, VIS, HALF)


@pytest.mark.parametrize("dx, dy, deg", [(120, -40, 15), (-100, 60, -30), (0, 0, 40)])
def test_finds_the_board_moved_and_turned_far_from_the_set_place(dx, dy, deg):
    found = find(scene(dx, dy, deg))
    assert found is not None and not found.lost and found.seen == 4
    # corners named after the set ones, so the plate's axes turn with the board
    assert np.abs(found.corners - truth(dx, dy, deg)).max() < 1.5


def test_corners_are_named_after_the_last_known_ones_not_the_set_place():
    # the board went on turning, from 40 to 50 degrees: past an eighth turn from the set place
    found = ColdFinder(VIS, HALF).find(scene(deg=-50), truth(deg=-40))
    assert found is not None and np.abs(found.corners - truth(deg=-50)).max() < 1.5


def test_a_dimmer_rim_round_the_board_still_gives_the_board_edge():
    # the step out of the rim is the strongest, so the first look starts 20 mm out and moves its
    # search; the second lands on the board's own edge
    img = paint(np.full((800, 1280), 60, np.uint8), at(SQUARE * (HALF + 20), 40, 20, 12), 160)
    found = find(paint(img, at(TRUE, 40, 20, 12), 210))
    assert found is not None and found.seen == 4
    assert np.abs(found.corners - truth(40, 20, 12)).max() < 1.5


def hidden_edge():
    """The board moved 60 mm, a sheet as bright as it against its +x edge."""
    img = scene(60, 0)
    cv2.rectangle(img, (int(at([(HALF, 0)], 60)[0, 0]), 100), (1230, 700), 210, -1)
    return img


def test_a_board_with_an_edge_hidden_is_not_guessed_from_the_other_three():
    # seen at a slant the board is no square in the calibration's mm: a fourth edge put a
    # board's width from the one across is several mm off
    assert find(hidden_edge()) is None


def test_an_edge_the_edge_tracker_cannot_fit_is_not_taken():
    # a ragged +x edge, three depths 3 mm apart in 7 px rows: the fine fit sees a step there, the
    # edge tracker cannot put half of it on one line and locks three edges only
    img = scene(60, 0)
    x = round(at([(HALF, 0.0)], 60)[0, 0])
    for i, y in enumerate(range(0, 800, 7)):
        img[y : y + 7, x : x + 6 * (i % 3)] = 210
    assert find(img) is None


def lamp(img, x, y, r=10, halo=20.0):
    """A lamp reflection: a white spot with a soft glow round it."""
    spot = np.zeros(img.shape, np.float32)
    cv2.circle(spot, (x, y), r, 1.0, -1)
    glow = cv2.GaussianBlur(spot, (0, 0), halo)
    glow = np.maximum(spot, 0.5 * glow / glow.max())
    return np.clip(img + glow * 255, 0, 255).astype(np.uint8)


def test_lamp_spots_and_ink_lines_do_not_stop_it():
    dx, dy, deg = 80, 30, -12
    img = scene(dx, dy, deg)
    edge = at([(HALF, 30.0), (-30.0, HALF), (50.0, -50.0)], dx, dy, deg).astype(int)
    for x, y in edge:
        img = lamp(img, x, y)
    # a thin line a little darker than the board and a dark marker line, both up to the edges
    a, b, c = at([(-HALF, -20.0), (HALF, 40.0), (20.0, -HALF)], dx, dy, deg).astype(int)
    cv2.line(img, tuple(a), tuple(b), 189, 4)
    cv2.line(img, tuple(c), tuple(a), 30, 3)
    found = find(img)
    assert found is not None and found.seen == 4
    assert np.abs(found.corners - truth(dx, dy, deg)).max() < 1.5


def a4_sheet():
    return paint(np.full((800, 1280), 60, np.uint8), at(SQUARE * (148.5, 105.0)), 210)


def table_noise():
    return np.clip(np.random.default_rng(1).normal(60, 6, (800, 1280)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("img", [table_noise(), np.zeros((800, 1280), np.uint8)])
def test_an_empty_table_gives_no_square_to_try(img):
    assert candidates(img) == [] and find(img) is None


def test_a_sheet_of_paper_alone_is_not_the_board():
    # its long sides are a board's width apart: squares across them get tried and refused
    assert find(a4_sheet()) is None


def test_a_board_close_to_the_frame_edge_is_found():
    # its bottom edge 5 mm from the frame's
    found = find(scene(0, 150))
    assert found is not None and np.abs(found.corners - truth(0, 150)).max() < 1.5


@pytest.mark.parametrize("dx, dy", [(0, 160), (0, 200), (380, 200), (-380, -230)])
def test_a_board_partly_out_of_the_frame_is_found_right_or_not_at_all(dx, dy):
    found = find(scene(dx, dy))
    assert found is None or np.abs(found.corners - truth(dx, dy)).max() < 1.5
