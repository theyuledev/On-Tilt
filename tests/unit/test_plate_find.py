import math

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision.plate_find import (
    SQUARE,
    PlateTracker,
    board_like,
    configured,
    crossings,
    edge_lines,
    fit_edges,
    steps,
)
from src.vision.px_to_mm import plate_mm

pytestmark = pytest.mark.unit

# a lens with no distortion looking straight down: 2 px per mm at the plate, image y down
CALIB = {
    "model": "standard",
    "K": [[800.0, 0.0, 640.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]],
    "D": [0.0] * 5,
    "H": [[400.0, 0.0, 0.0], [0.0, -400.0, 0.0], [0.0, 0.0, 1.0]],
}
VIS = {**load()["vision"], "calib": CALIB}
HALF = 120.0
TRUE = HALF * SQUARE
SPOTS = [(60.0, -40.0), (-100.0, 100.0), (0.0, 0.0)]


def at(mm, dx=0.0, dy=0.0, deg=0.0) -> np.ndarray:
    """Picture px of board mm, the board moved dx, dy px and turned about its middle."""
    a = math.radians(deg)
    turn = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (np.asarray(mm, float) * (2, -2)) @ turn.T + (640 + dx, 400 + dy)


def paint(img: np.ndarray, corners_px: np.ndarray, level: float) -> np.ndarray:
    """A convex quad with each edge pixel covered by the share of it the quad takes, so the
    drawn edge sits where the corners say to a small fraction of a pixel."""
    y, x = np.mgrid[: img.shape[0], : img.shape[1]]
    cover = np.ones(img.shape)
    for p, q in zip(corners_px, np.roll(corners_px, -1, axis=0), strict=True):
        n = np.array([q[1] - p[1], p[0] - q[0]]) / math.dist(p, q)
        if n @ (corners_px.mean(0) - p) > 0:
            n = -n
        cover *= np.clip(0.5 - ((x - p[0]) * n[0] + (y - p[1]) * n[1]), 0, 1)
    return (img + (level - img) * cover).astype(np.uint8)


def scene(dx=0.0, dy=0.0, deg=0.0, half=(HALF, HALF), level=210, table=60) -> np.ndarray:
    img = np.full((800, 1280), table, np.uint8)
    return paint(img, at(SQUARE * half, dx, dy, deg), level)


def tracker(image=None) -> PlateTracker:
    tr = PlateTracker(VIS, HALF, configured(HALF, VIS))
    tr.update(scene() if image is None else image)
    return tr


def read(tr, px):
    return plate_mm(*px, tr.mapped())


def off_by(tr, dx=0.0, dy=0.0, deg=0.0) -> float:
    """Worst mm error of fixed board spots, read through the tracker's map."""
    return max(math.dist(read(tr, at(s, dx, dy, deg)), s) for s in SPOTS)


def test_configured_corners_are_the_settings_centre_and_yaw():
    cfg = {**VIS, "calib_centre_mm": [10.0, -5.0], "calib_yaw_deg": 90.0}
    assert configured(HALF, VIS) == pytest.approx(TRUE)
    assert configured(HALF, cfg) == pytest.approx(np.roll(TRUE, 1, axis=0) + (10.0, -5.0))


def test_lands_on_the_board_from_the_corners_it_is_handed():
    start = configured(HALF, {**VIS, "calib_centre_mm": [6.0, -5.0], "calib_yaw_deg": 2.0})
    tr = PlateTracker(VIS, HALF, start)
    assert tr.lost
    tr.update(scene())
    assert not tr.lost
    assert np.abs(tr.corners - TRUE).max() < 0.15


def test_follows_a_shift_and_a_turn_in_one_frame():
    tr = tracker()
    assert off_by(tr) < 0.1
    tr.update(scene(10, -6, 1.0))
    assert not tr.lost and off_by(tr, 10, -6, 1.0) < 0.1
    # the map it started with would read the same spots this far out
    assert off_by(tracker(), 10, -6, 1.0) > 4.0


def test_a_shaking_camera_stays_out_of_the_mm():
    tr, still = tracker(), tracker()
    rng = np.random.default_rng(3)
    worst = fixed = 0.0
    for dx, dy in rng.uniform(-2, 2, (30, 2)):
        tr.update(scene(dx, dy))
        worst = max(worst, off_by(tr, dx, dy))
        fixed = max(fixed, off_by(still, dx, dy))
    assert worst < 0.1 and fixed > 0.8


@pytest.mark.parametrize("shade", [30, 110, 255])
def test_something_over_part_of_an_edge_does_not_move_the_corners(shade):
    img = scene()
    # 40% of the top edge, reaching well inside and outside the board
    cv2.rectangle(img, (520, 100), (712, 230), shade, -1)
    tr = tracker(img)
    assert not tr.lost
    assert np.abs(tr.corners - TRUE).max() < 0.1


def test_a_hidden_edge_keeps_its_line_while_the_others_follow():
    def covered(dx):
        img = scene(dx)
        cv2.rectangle(img, (780, 100), (1000, 700), 110, -1)
        return img

    tr = tracker()
    tr.update(scene(4))
    assert tr.corners[:, 0] == pytest.approx(TRUE[:, 0] + 2.0, abs=0.1)
    tr.update(covered(7))
    assert not tr.lost
    x = tr.corners[:, 0]
    # the +x edge is under the cover and stays where it was last seen, the -x edge goes on
    assert x[[0, 3]] == pytest.approx(HALF + 2.0, abs=0.1)
    assert x[[1, 2]] == pytest.approx(-HALF + 3.5, abs=0.1)


@pytest.mark.parametrize("fill", [0, 60, 210])
def test_no_board_keeps_the_corners_and_says_lost(fill):
    tr = tracker()
    before = tr.corners.copy()
    tr.update(np.full((800, 1280), fill, np.uint8))
    # a plain bool: it goes out to the page as json
    assert tr.lost is True and np.array_equal(tr.corners, before)
    tr.update(scene())
    assert tr.lost is False


@pytest.mark.parametrize(
    "half", [(HALF * 1.08, HALF * 1.08), (HALF * 1.08, HALF), (HALF, HALF * 0.93)]
)
def test_a_square_of_the_wrong_size_or_shape_is_refused(half):
    tr = tracker()
    before = tr.corners.copy()
    tr.update(scene(half=half))
    assert tr.lost and np.array_equal(tr.corners, before)


def test_the_edge_is_the_first_drop_when_a_dimmer_strip_lies_past_it():
    img = np.full((800, 1280), 60, np.uint8)
    # 7 mm of something dimmer sticking out under the +x side, a far bigger step at its end
    cv2.rectangle(img, (880, 160), (894, 640), 170, -1)
    tr = tracker(paint(img, at(TRUE), 210))
    assert not tr.lost
    assert np.abs(tr.corners - TRUE).max() < 0.15


def test_the_search_moves_with_the_board_once_it_is_off_the_middle():
    tr = tracker()
    grid = tr.grid
    for step in range(1, 9):
        tr.update(scene(dx=8 * step))
        assert not tr.lost and off_by(tr, dx=8 * step) < 0.1
    # 32 mm in all, past the 20 mm reach of the first search
    assert tr.grid is not grid


def profile(edge_mm, across, inner=200.0, outer=80.0):
    """A soft step: each sample is the mean over its own width."""
    d = across[1] - across[0]
    cover = np.clip((edge_mm - (across - d / 2)) / d, 0, 1)
    return outer + (inner - outer) * cover


def test_steps_find_the_drop_to_a_fraction_of_a_sample():
    across = np.arange(-20, 20, 1 / VIS["px_per_mm"])
    edges_mm = np.array([-7.3, -0.1, 0.0, 0.17, 4.44, 11.0])
    cut = np.array([profile(e, across) for e in edges_mm])[None]
    assert steps(cut, across, VIS)[0] == pytest.approx(edges_mm, abs=0.06)


def test_steps_take_the_first_drop_and_say_nan_on_a_flat_profile():
    across = np.arange(-20, 20, 1 / VIS["px_per_mm"])
    two = profile(-3.0, across, 200, 165) + profile(9.0, across, 165, 40) - 165
    flat = np.full_like(across, 150.0)
    # 4% over the whole profile is shading, not an edge
    ramp = np.linspace(150, 144, len(across))
    got = steps(np.array([[two, flat, ramp]]), across, VIS)[0]
    assert got[0] == pytest.approx(-3.0, abs=0.06)
    assert np.isnan(got[1]) and np.isnan(got[2])


def test_fit_ignores_the_steps_off_the_line_and_needs_half_of_them():
    along = np.tile(np.linspace(-100, 100, 20), (4, 1))
    off = 2.0 + 0.01 * along
    off[0, 3:11] = [9.0, -6.0, np.nan, 14.0, 9.5, np.nan, -12.0, 5.0]
    off[1, :11] = np.nan
    off[2, 5:16] += [4.0, -7.0, 9.0, 5.0, -3.5, 12.0, -9.0, 6.0, 15.0, -5.0, 8.0]
    out, slope, seen = fit_edges(along, off, VIS)
    assert seen.tolist() == [True, False, False, True]
    assert out[[0, 3]] == pytest.approx(2.0, abs=1e-6)
    assert slope[[0, 3]] == pytest.approx(0.01, abs=1e-6)


def test_lines_cross_at_the_corners_and_a_fit_moves_its_edge():
    assert crossings(edge_lines(TRUE)) == pytest.approx(TRUE)
    # the +y edge 5 mm further out: its two corners follow, the others stay
    moved = crossings(edge_lines(TRUE, [5.0, 0, 0, 0], [0.0, 0, 0, 0]))
    assert moved == pytest.approx(TRUE + [(0, 5), (0, 5), (0, 0), (0, 0)])


def seen_from(tilt_deg, turn_deg, shift_mm):
    """Where the calibration puts the corners of a real square that is turned in its plane,
    tipped about its x axis and slid along it. The lens is 400 mm above the sheet's plane."""
    a, b = math.radians(turn_deg), math.radians(tilt_deg)
    x, y = (TRUE @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])).T
    x, y, lift = x + shift_mm, y * math.cos(b), y * math.sin(b)
    return np.column_stack([x, y]) * (400 / (400 - lift))[:, None]


def test_board_like_takes_a_real_square_at_any_place_turn_and_tilt():
    assert board_like(TRUE, HALF, VIS)
    assert board_like(seen_from(0, 37, 60), HALF, VIS)
    tipped = seen_from(6, 20, -30)
    # not a square in the calibration's mm any more, but still the picture of one
    sides = np.linalg.norm(np.roll(tipped, -1, axis=0) - tipped, axis=1)
    assert np.ptp(sides) > 5 and board_like(tipped, HALF, VIS)
    assert not board_like(TRUE * 1.05, HALF, VIS)
    assert not board_like(TRUE * (1.05, 1.0), HALF, VIS)
    assert not board_like(TRUE + [(8, 0), (0, 0), (0, 0), (8, 0)], HALF, VIS)
    assert not board_like(np.zeros((4, 2)), HALF, VIS)
