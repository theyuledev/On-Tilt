import json

import cv2
import numpy as np
import pytest

from src.vision.calib import project, undistort
from src.vision.markers import board_pose, detector, find, load_board, relative_tilt
from tests.unit.test_plate_pose import STANDARD, T, pose, rot

pytestmark = pytest.mark.unit

CFG = {
    "marker_dict": "DICT_4X4_50",
    "marker_refine_win_px": 5,
    "marker_thresh_win_px": 23,
    "marker_min_perimeter_rate": 0.03,
}
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# 160 px print plus a 40 px quiet zone, 75 mm wide; its outer edge sits half a pixel outside
# the corner pixels, so print px = (mm from the marker centre) / 75 * 240 + 119.5
PRINT_MM = 75.0


def marker(cx, cy, h=25.0):
    pts = [[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]]
    return np.c_[np.array(pts, np.float32), np.zeros(4, np.float32)]


PLATE = {0: marker(-115, 115), 1: marker(115, 115), 2: marker(115, -115), 3: marker(-115, -115)}
BASE = {4: marker(-60, 0), 5: marker(60, 0)}
R_BASE = pose()[0]
# below-left of the diamond: straight below it sits on plate id 2 and runs past row 800
T_BASE = T + [-200, 170, 30]


def truth(pts, r, t, lens=STANDARD):
    return project(lens, pts, cv2.Rodrigues(r)[0], t)


def paste(frame, i, pts, r, t, lens):
    """Each pixel's ray through the lens meets the board plane, so the print lands exactly
    where project() puts it, lens distortion included."""
    img = cv2.aruco.generateImageMarker(DICT, i, 160)
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    quad = truth(pts.mean(0) + (pts - pts.mean(0)) * 1.5, r, t, lens)
    x0, y0 = np.maximum(np.floor(quad.min(0)).astype(int) - 2, 0)
    x1, y1 = np.minimum(np.ceil(quad.max(0)).astype(int) + 3, frame.shape[::-1])
    ys, xs = np.mgrid[y0:y1, x0:x1]
    rays = undistort(lens, np.c_[xs.ravel(), ys.ravel()]).reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(rays, np.linalg.inv(np.c_[r[:, :2], t])).reshape(xs.shape + (2,))
    sx = (uv[..., 0] - pts[0, 0] - 25) / PRINT_MM * 240 + 119.5
    sy = (pts[0, 1] - 25 - uv[..., 1]) / PRINT_MM * 240 + 119.5
    patch = cv2.remap(img, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR)
    inside = (sx >= -0.5) & (sx <= 239.5) & (sy >= -0.5) & (sy <= 239.5)
    frame[y0:y1, x0:x1][inside] = patch[inside]


def render(poses, seed=0, lens=STANDARD):
    """Prints seen through the lens, then 0.7 px blur and gray noise sigma 3."""
    frame = np.full((800, 1280), 200, np.uint8)
    for board, r, t in poses:
        for i, pts in board.items():
            paste(frame, i, pts, r, t, lens)
    frame = cv2.GaussianBlur(frame, (0, 0), 0.7)
    noise = np.random.default_rng(seed).normal(0, 3, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def solved(found, board, ids, **kw):
    out = board_pose({i: found[i] for i in ids}, board, STANDARD, **kw)
    return cv2.Rodrigues(out[0])[0], out[1].ravel()


def test_finds_all_six_markers_to_a_third_of_a_pixel():
    r = pose(tip_y=4, tip_x=-3)[0]
    px = np.concatenate([truth(BASE[i], R_BASE, T_BASE) for i in BASE])
    assert px.min() > 0 and px[:, 0].max() < 1280 and px[:, 1].max() < 800
    found = find(detector(CFG), render([(PLATE, r, T), (BASE, R_BASE, T_BASE)]))
    assert sorted(found) == [0, 1, 2, 3, 4, 5]
    err = [found[i] - truth(PLATE[i], r, T) for i in PLATE]
    err += [found[i] - truth(BASE[i], R_BASE, T_BASE) for i in BASE]
    err = np.concatenate(err)
    assert np.sqrt((err**2).sum(1).mean()) < 0.3


def test_tilt_from_four_three_and_two_markers():
    det = detector(CFG)
    for k, (tip_y, tip_x) in enumerate(np.random.default_rng(0).uniform(-8, 8, (20, 2))):
        r, tx, ty = pose(tip_y, tip_x)
        found = find(det, render([(PLATE, r, T)], seed=k))
        # ids 1, 2 share a diagonal, so tilt about it rests on each marker's own 50 mm
        for ids, bar in (([0, 1, 2, 3], 0.2), ([1, 2, 3], 0.2), ([1, 2], 0.6)):
            r_got, t_got = solved(found, PLATE, ids)
            got = relative_tilt(r_got)
            assert abs(got[0] - tx) <= bar and abs(got[1] - ty) <= bar, ids
            assert np.linalg.norm(t_got - T) < 2, ids


def test_two_markers_with_a_guess_refine_from_it():
    r, tx, ty = pose(tip_y=5, tip_x=-5)
    found = find(detector(CFG), render([(PLATE, r, T)]))
    guess = (cv2.Rodrigues(rot([0, 1, 0], 1) @ r)[0], T + [3, -2, 5])
    got = relative_tilt(solved(found, PLATE, [1, 2], guess=guess)[0])
    assert abs(got[0] - tx) <= 0.2 and abs(got[1] - ty) <= 0.2


def test_single_marker_needs_min_count_one_and_follows_the_guess():
    r, tx, ty = pose(tip_y=5, tip_x=5)
    found = find(detector(CFG), render([(PLATE, r, T)]))
    assert board_pose({0: found[0]}, PLATE, STANDARD) is None
    guess = (cv2.Rodrigues(rot([1, 0, 0], 1) @ r)[0], T)
    # one 50 mm marker at 380 mm with ~0.2 px corner noise: measured up to 0.93 deg off
    for i in PLATE:
        got = relative_tilt(solved(found, PLATE, [i], guess=guess, min_count=1)[0])
        assert abs(got[0] - tx) <= 1.0 and abs(got[1] - ty) <= 1.0, i


def test_tilt_sign_matches_pose_on_both_paths():
    for tip in ({"tip_y": 6}, {"tip_x": 6}):
        r, tx, ty = pose(**tip)
        raw, rel = relative_tilt(r), relative_tilt(r, R_BASE)
        assert raw == pytest.approx((tx, ty), abs=1e-9)
        assert rel == pytest.approx((tx, ty), abs=0.2)
        assert min(abs(tx), abs(ty)) < 1e-9 < max(abs(tx), abs(ty))


def test_left_handed_board_points_put_z_away_from_the_camera():
    r = pose(tip_y=4, tip_x=-3)[0]
    found = find(detector(CFG), render([(PLATE, r, T)]))
    flipped = {i: pts * np.float32([1, -1, 1]) for i, pts in PLATE.items()}
    assert solved(found, PLATE, PLATE)[0][2, 2] < 0
    assert solved(found, flipped, PLATE)[0][2, 2] > 0


def test_base_board_cancels_a_stand_shake_under_render_noise():
    det = detector(CFG)
    r = pose(tip_y=3, tip_x=-2)[0]
    tilts = []
    for s, seed in ((np.eye(3), 1), (rot([1, 0, 0], 2), 2)):
        found = find(det, render([(PLATE, s @ r, s @ T), (BASE, s @ R_BASE, s @ T_BASE)], seed))
        rp, rb = solved(found, PLATE, PLATE)[0], solved(found, BASE, BASE)[0]
        tilts.append((relative_tilt(rp), relative_tilt(rp, rb)))
    raw = np.abs(np.subtract(tilts[0][0], tilts[1][0]))
    rel = np.abs(np.subtract(tilts[0][1], tilts[1][1]))
    assert raw.max() > 1.0  # a 2 deg nod shows up as ~1.4 deg on each diamond axis
    assert rel.max() < 0.3


def test_rois_give_the_full_frame_corners():
    r = pose(tip_y=4, tip_x=-3)[0]
    gray = render([(PLATE, r, T), (BASE, R_BASE, T_BASE)])
    det = detector(CFG)
    full = find(det, gray)
    rois = {}
    for i, c in full.items():
        pad = 1.5 * np.linalg.norm(c[1] - c[0])
        rois[i] = (*(c.min(0) - pad), *(c.max(0) + pad))
    boxed = find(det, gray, rois)
    assert sorted(boxed) == sorted(full)
    assert max(np.abs(boxed[i] - full[i]).max() for i in full) < 0.05


def test_load_board_round_trips(tmp_path):
    raw = {"plate": {str(i): p.tolist() for i, p in PLATE.items()}}
    raw["base"] = {str(i): p.tolist() for i, p in BASE.items()}
    (tmp_path / "board.json").write_text(json.dumps(raw))
    plate, base = load_board(tmp_path / "board.json")
    assert plate.keys() == PLATE.keys() and base.keys() == BASE.keys()
    assert all(np.array_equal(plate[i], PLATE[i]) for i in PLATE)
    assert all(base[i].dtype == np.float32 and base[i].shape == (4, 3) for i in BASE)
