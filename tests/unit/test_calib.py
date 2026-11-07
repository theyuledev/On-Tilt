import itertools
import re
from datetime import date

import cv2
import numpy as np
import pytest

from src.vision import calib as cb
from tools import calibrate

pytestmark = pytest.mark.unit

PATTERN, SQ, SIZE = (7, 8), 25.0, (1280, 800)
FISHEYE = {
    "model": "fisheye",
    "K": [[420, 0, 650], [0, 420, 390], [0, 0, 1]],
    "D": [0.06, -0.03, 0.01, -0.002],
}
STANDARD = {
    "model": "standard",
    "K": [[720, 0, 630], [0, 720, 410], [0, 0, 1]],
    "D": [-0.32, 0.11, 0.002, -0.0015, -0.018, 0, 0, 0],
}
PINHOLE = {"model": "standard", "K": [[400, 0, 320], [0, 400, 200], [0, 0, 1]], "D": [0] * 5}
VISION = {"board_corners": [7, 8], "board_square_mm": SQ, "calib_views": 4, "calib_timeout_s": 60}
OBJ = cb.board_points(PATTERN, SQ)
MID = OBJ.mean(axis=0)


def pose(at, rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return np.asarray(rvec, float), np.asarray(at, float) - R @ MID


def views(cam, noise=0.0):
    rng = np.random.default_rng(1)
    out = []
    for x, y, tilt in itertools.product(
        (-300, 0, 300), (-180, 180), ((0.35, 0.0, 0.3), (-0.2, 0.3, -0.5))
    ):
        px = cb.project(cam, OBJ, *pose((x, y, 450), tilt))
        if ((px >= 0) & (px < SIZE)).all():
            out.append(px + rng.normal(0, noise, px.shape))
    return out


def render(rvec, tvec, cam=PINHOLE, size=(640, 400), plate=265.0):
    """Sheet on a bright plate on a dark table, drawn at 0.8 px/mm in sheet mm."""
    k, lo = 0.8, MID[:2] - plate / 2 - 20
    canvas = np.full((round((plate + 40) * k),) * 2, 60, np.uint8)

    def fill(x0, x1, y0, y1, value):
        (c0, r0), (c1, r1) = np.round((np.array([[x0, y0], [x1, y1]]) - lo) * k).astype(int)
        canvas[r0:r1, c0:c1] = value

    fill(*(MID[0] + np.array([-1, 1]) * plate / 2), *(MID[1] + np.array([-1, 1]) * plate / 2), 200)
    fill(-40, 190, -40, 215, 255)
    for r, c in itertools.product(range(9), range(8)):
        if (r + c) % 2:
            fill((c - 1) * SQ, c * SQ, (r - 1) * SQ, r * SQ, 0)
    to_mm = np.array([[1 / k, 0, lo[0] + 0.5 / k], [0, 1 / k, lo[1] + 0.5 / k], [0, 0, 1]])
    R, _ = cv2.Rodrigues(rvec)
    plane = np.array(cam["K"], float) @ np.column_stack([R[:, 0], R[:, 1], tvec])
    image = cv2.warpPerspective(canvas, plane @ to_mm, size, borderValue=60)
    return cv2.GaussianBlur(image, (3, 3), 0.7), cb.project(cam, OBJ, rvec, tvec)


def jpg(image):
    return cv2.imencode(".jpg", image)[1].tobytes()


def frames(*jpgs):
    it = iter(jpgs)
    return lambda: next(it)


def test_finds_corners_in_sheet_order_on_a_rendered_board():
    image, truth = render(*pose((20, -10, 450), (0.25, -0.15, 3.5)))
    found = cb.find_corners(image, PATTERN)
    assert found is not None
    assert np.abs(found - truth).max() < 0.25
    assert cb.find_corners(np.full_like(image, 128), PATTERN) is None


@pytest.mark.parametrize("cam", [FISHEYE, STANDARD], ids=lambda c: c["model"])
def test_fit_picks_the_lens_model_that_made_the_views(cam):
    fits = cb.fit_lens(views(cam), PATTERN, SQ, SIZE)
    assert len(fits) == 2
    assert fits[0]["model"] == cam["model"]
    assert fits[0]["rms_px"] < 1e-3
    assert np.allclose(fits[0]["K"], cam["K"], atol=0.5)


@pytest.mark.parametrize("cam", [FISHEYE, STANDARD], ids=lambda c: c["model"])
def test_plate_points_map_to_mm_and_back(cam):
    lens = cb.fit_lens(views(cam, noise=0.1), PATTERN, SQ, SIZE)[0]
    flat = pose((120, 20, 470), (0.03, -0.02, np.pi / 4))
    corners = cb.project(cam, OBJ, *flat)
    noisy = corners + np.random.default_rng(2).normal(0, 0.1, corners.shape)
    calib = cb.fit_plate(lens, noisy, PATTERN, SQ)
    assert calib["plate_rms_mm"] < 0.5

    plate_mm = np.array([[0, 0], [60, -40], [120, 120], [-125, 90]], float)
    on_sheet = np.column_stack([MID[0] + plate_mm[:, 0], MID[1] - plate_mm[:, 1], np.zeros(4)])
    px = cb.project(cam, on_sheet, *flat)
    assert np.abs(cb.px_to_plate_mm(px, calib) - plate_mm).max() < 0.5
    assert np.abs(cb.plate_mm_to_px(plate_mm, calib) - px).max() < 0.5


@pytest.mark.parametrize("mirrored", [False, True])
def test_plate_frame_turns_counterclockwise_in_the_image(mirrored):
    corners = cb.project(STANDARD, OBJ, *pose((0, 0, 450), (0.1, 0, 0.7)))
    if mirrored:
        corners = corners.reshape(8, 7, 2)[:, ::-1].reshape(-1, 2)
    calib = cb.fit_plate(STANDARD, corners, PATTERN, SQ)
    o, x, y = cb.plate_mm_to_px([[0, 0], [10, 0], [0, 10]], calib)
    a, b = x - o, y - o
    assert a[0] * b[1] - a[1] * b[0] < 0
    assert calib["plate_rms_mm"] < 1e-3


def test_new_view_needs_the_board_moved_more_than_a_square():
    corners = cb.project(STANDARD, OBJ, *pose((0, 0, 450), (0, 0, 0)))
    step = cb.square_px(corners, PATTERN)
    assert cb.is_new_view(corners, [], PATTERN)
    assert not cb.is_new_view(corners + [0.8 * step, 0], [corners], PATTERN)
    assert cb.is_new_view(corners + [1.2 * step, 0], [corners], PATTERN)
    moved = [corners, corners + [2.5 * step, 0]]
    assert not cb.is_new_view(corners + [3 * step, 0], moved, PATTERN)


@pytest.mark.parametrize("plate", [265.0, 305.0])
def test_measures_the_plate_edges_around_the_sheet(plate):
    flat = pose((10, 0, 550), (0.02, 0, np.pi / 4))
    image, corners = render(*flat, plate=plate)
    calib = cb.fit_plate(PINHOLE, corners, PATTERN, SQ)
    assert cb.plate_size_mm(image, calib) == pytest.approx((plate, plate), abs=3)
    assert cb.plate_size_mm(np.zeros_like(image), calib) is None


def test_calib_json_round_trip(tmp_path):
    corners = cb.project(STANDARD, OBJ, *pose((50, 0, 450), (0.1, 0, 0.7)))
    calib = cb.fit_plate(STANDARD, corners, PATTERN, SQ)
    cb.save_calib(calib, tmp_path / "calib.json")
    back = cb.load_calib(tmp_path / "calib.json")
    assert back == calib
    pts = [[10, 20], [-50, 30]]
    assert np.allclose(cb.px_to_plate_mm(cb.plate_mm_to_px(pts, back), back), pts)


def test_capture_keeps_moved_views_then_fit_writes_the_lens(tmp_path):
    views_dir = tmp_path / "views"
    views_dir.mkdir()
    (views_dir / "99.jpg").write_bytes(b"old run")
    a, b, c, d = (
        jpg(render(*pose(at, tilt))[0])
        for at, tilt in [
            ((-120, 0, 450), (0.3, 0, 0)),
            ((0, 0, 450), (0, 0.3, 0.5)),
            ((120, 0, 450), (-0.3, 0.1, 1.0)),
            ((0, 0, 500), (0.1, -0.3, 2.0)),
        ]
    )
    blank = jpg(np.full((400, 640), 128, np.uint8))
    assert calibrate.capture(VISION, frames(a, a, blank, b, c, c, d), views_dir) == 4
    assert sorted(p.name for p in views_dir.glob("*.jpg")) == [f"0{i}.jpg" for i in range(1, 5)]
    assert (views_dir / "03.jpg").read_bytes() == c

    lens = calibrate.fit(VISION, views_dir, tmp_path / "calib.json")
    assert lens["rms_px"] < 0.5
    assert cb.load_calib(tmp_path / "calib.json") == lens


def test_capture_stops_at_the_time_limit(tmp_path):
    assert calibrate.capture({**VISION, "calib_timeout_s": 0}, frames(), tmp_path) == 0


def test_lens_set_writes_a_measured_standard_lens(tmp_path):
    out = tmp_path / "calib.json"
    with pytest.raises(SystemExit):
        calibrate.lens_set(["700", "700"], out)
    lens = calibrate.lens_set("700 690 650 420 -0.3 0.1 0.001 -0.002 -0.02".split(), out)
    assert cb.load_calib(out) == lens
    assert lens["model"] == "standard"
    assert lens["K"] == [[700, 0, 650], [0, 690, 420], [0, 0, 1]]
    assert lens["D"] == [-0.3, 0.1, 0.001, -0.002, -0.02]
    assert lens["measured"] == date.today().isoformat()
    assert np.allclose(cb.undistort(lens, [[650, 420]]), 0)


def test_plate_step_adds_the_homography_to_the_lens(tmp_path, capsys):
    out = tmp_path / "calib.json"
    with pytest.raises(SystemExit):
        calibrate.plate(VISION, frames(), out)
    cb.save_calib(PINHOLE, out)
    flat = pose((10, 0, 450), (0.02, 0, np.pi / 4))
    image, _ = render(*flat)
    calib = calibrate.plate(VISION, frames(jpg(np.zeros_like(image)), jpg(image)), out)
    assert cb.load_calib(out) == calib
    assert calib["measured"] == date.today().isoformat()
    assert calib["plate_rms_mm"] < 0.5
    assert np.abs(cb.plate_mm_to_px([[0, 0]], calib) - cb.project(PINHOLE, MID, *flat)).max() < 0.5
    edges = re.search(r"plate edges (\d+) x (\d+) mm", capsys.readouterr().out).groups()
    assert [float(e) for e in edges] == pytest.approx([265, 265], abs=3)
