import json
import math

import cv2
import numpy as np
import pytest

from src.vision.calib import project
from src.vision.markers import detector, find, load_board
from tests.unit.test_plate_pose import STANDARD, T, rot
from tests.vision.test_markers import BASE, CFG, DICT, PLATE, marker, render
from tools.mark_board import base_board, fit_plate, sheet_spacing_mm

pytestmark = pytest.mark.unit

PLATE_MM, SIZE, YAW = 305.0, 50.0, 30.0
INSETS = {i: (12.5, 12.5) for i in PLATE}
TURNED, TURN_DEG = 3, 3.0
C3 = np.array([-115.0, -115.0])


def facing(yaw):
    """Plate under the cam tipped 2 deg, its +x at -yaw in the y-up image, so the flat route
    with plate_yaw_deg = yaw reads it as +x."""
    return np.diag([1.0, -1.0, -1.0]) @ rot([0, 0, 1], -yaw) @ rot([0, 1, 0], 2)


def truth():
    xy = C3 + marker(0, 0)[:, :2] @ rot([0, 0, 1], TURN_DEG)[:2, :2].T
    return {**PLATE, TURNED: np.c_[xy, np.zeros(4)]}


def seen(yaw=YAW):
    """id 3 stuck on 3 deg counterclockwise, the rest square to the plate."""
    r = facing(yaw)
    rest = {i: PLATE[i] for i in PLATE if i != TURNED}
    turned = ({TURNED: marker(0, 0)}, r @ rot([0, 0, 1], TURN_DEG), r @ np.append(C3, 0) + T)
    return find(detector(CFG), render([(rest, r, T), turned]))


def fit(found, yaw=YAW):
    return fit_plate(found, STANDARD, [0, 1, 2, 3], INSETS, SIZE, PLATE_MM, yaw)


def centres(out):
    return {int(i): np.asarray(p)[:, :2].mean(0) for i, p in out["plate"].items()}


def test_board_json_round_trips_within_a_fifth_of_a_mm(tmp_path):
    out = fit(seen(), yaw=YAW - 5)
    out |= {"base": base_board([4, 5], SIZE, 120.0), "marker_size_mm": SIZE}
    path = tmp_path / "board.json"
    path.write_text(json.dumps(out))
    plate, base = load_board(path)
    for i, pts in truth().items():
        assert np.abs(plate[i] - pts).max() < 0.2, i
    for i in BASE:
        assert np.abs(base[i] - BASE[i]).max() < 1e-4, i
    for i in PLATE:
        want = TURN_DEG if i == TURNED else 0.0
        assert out["marker_rot_deg"][str(i)] == pytest.approx(want, abs=0.3), i
        assert out["marker_seen_mm"][str(i)] == pytest.approx(SIZE, abs=0.5), i
    assert out["plate_corner"] == {"0": "TL", "1": "TR", "2": "BR", "3": "BL"}
    rvec = cv2.Rodrigues(facing(YAW))[0]
    px = np.array([project(STANDARD, PLATE[i], rvec, T).mean(0) for i in PLATE])
    sides = np.linalg.norm(np.roll(px, -1, 0) - px, axis=1)
    assert out["square_side_px"] == pytest.approx(sides.mean(), abs=0.5)
    assert out["px_per_mm_at_plate"] == pytest.approx(sides.mean() / 230, rel=0.005)


def test_board_x_lies_along_the_flat_route_x():
    found = seen()
    out = fit(found, yaw=YAW + 10)
    c = centres(out)
    bl, br = (i for n in ("BL", "BR") for i in c if out["plate_corner"][str(i)] == n)
    du, dv = (found[br].mean(0) - found[bl].mean(0)) * [1, -1]
    a = math.radians(YAW)
    x, y = du * math.cos(a) - dv * math.sin(a), du * math.sin(a) + dv * math.cos(a)
    assert math.degrees(math.atan2(y, x)) == pytest.approx(0, abs=3)


def test_a_quarter_turn_more_yaw_moves_every_id_one_corner_on():
    found = seen()
    was, now = centres(fit(found)), centres(fit(found, yaw=YAW + 90))
    for i in PLATE:
        assert now[i] == pytest.approx([-was[i][1], was[i][0]], abs=1e-6), i


def test_sheet_spacing_from_the_marker_size():
    sheet = np.full((400, 900), 255, np.uint8)
    for i, x0 in ((4, 100), (5, 600)):
        sheet[100:300, x0 : x0 + 200] = cv2.aruco.generateImageMarker(DICT, i, 200)
    got = sheet_spacing_mm(sheet, detector(CFG), [4, 5], SIZE)
    assert got == pytest.approx(500 / 200 * SIZE, abs=0.1)


def test_missing_plate_ids_are_named():
    found = seen()
    del found[2]
    with pytest.raises(ValueError, match=r"not seen: \[2\]"):
        fit(found)
