"""ArUco boards: plate markers (ids 0-3) and the table base (ids 4-5).

Board frame: +x right, +y up, +z toward the camera. A marker centred at (cx, cy) with half
size h has object points TL (cx-h, cy+h), TR (cx+h, cy+h), BR (cx+h, cy-h), BL (cx-h, cy-h),
the pattern's own TL/TR/BR/BL when the print lies with its top edge toward +y.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.io.settings import ROOT
from src.vision.calib import undistort

DICTS = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50}
EYE = np.eye(3)


def detector(cfg: dict) -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = cfg["marker_refine_win_px"]
    p.adaptiveThreshWinSizeMin = p.adaptiveThreshWinSizeMax = cfg["marker_thresh_win_px"]
    p.minMarkerPerimeterRate = cfg["marker_min_perimeter_rate"]
    # a quiet zone touching the frame edge must not drop the marker inside it
    p.minDistanceToBorder = 0
    words = cv2.aruco.getPredefinedDictionary(DICTS[cfg["marker_dict"]])
    return cv2.aruco.ArucoDetector(words, p)


def _detect(det, gray, offset=(0, 0)) -> dict[int, np.ndarray]:
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    shift = np.asarray(offset, np.float32)
    return {int(i): c.reshape(4, 2) + shift for i, c in zip(ids.ravel(), corners, strict=True)}


def find(det, gray: np.ndarray, rois: dict | None = None) -> dict[int, np.ndarray]:
    """Corners (4, 2) float32 per id, TL TR BR BL of the pattern, in full-frame px.
    rois: {id: (x0, y0, x1, y1)} boxes to search instead of the whole frame."""
    if rois is None:
        return _detect(det, gray)
    h, w = gray.shape[:2]
    found = {}
    for x0, y0, x1, y1 in rois.values():
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1) + 1, w), min(int(y1) + 1, h)
        if x1 > x0 and y1 > y0:
            found.update(_detect(det, gray[y0:y1, x0:x1], (x0, y0)))
    return found


def _nearest(rvecs, tvecs, guess):
    r_guess = cv2.Rodrigues(np.asarray(guess[0], np.float64))[0]

    def angle(rvec):
        return np.linalg.norm(cv2.Rodrigues(r_guess.T @ cv2.Rodrigues(rvec)[0])[0])

    k = min(range(len(rvecs)), key=lambda k: angle(rvecs[k]))
    return rvecs[k], tvecs[k]


def board_pose(corners, board, lens, guess=None, min_count=2):
    """(rvec, tvec) of the board from the markers seen, or None with fewer than min_count.
    One solve over every corner; a lone marker needs a guess to pick the IPPE branch."""
    ids = sorted(set(corners) & set(board))
    if len(ids) < min_count:
        return None
    obj = np.concatenate([board[i] for i in ids]).astype(np.float64).reshape(-1, 1, 3)
    img = undistort(lens, np.concatenate([corners[i] for i in ids])).reshape(-1, 1, 2)
    if len(ids) == 1:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, EYE, None, flags=cv2.SOLVEPNP_IPPE)
        return _nearest(rvecs, tvecs, guess) if guess is not None else (rvecs[0], tvecs[0])
    if guess is None:
        _, rvec, tvec = cv2.solvePnP(obj, img, EYE, None, flags=cv2.SOLVEPNP_IPPE)
    else:
        rvec, tvec = (np.array(v, np.float64).reshape(3, 1) for v in guess)
    return cv2.solvePnPRefineLM(obj, img, EYE, None, rvec, tvec)


def relative_tilt(r_plate, r_base=None) -> tuple[float, float]:
    """Degrees the plate's +x and +y edges drop below level: level is the camera looking
    straight down, or the base board's plane when given."""
    down = np.array([0.0, 0.0, 1.0]) if r_base is None else -np.asarray(r_base)[:, 2]
    tilt = np.degrees(np.arcsin(np.clip(down @ np.asarray(r_plate)[:, :2], -1, 1)))
    return float(tilt[0]), float(tilt[1])


def _board(raw: dict) -> dict[int, np.ndarray]:
    return {int(k): np.asarray(v, np.float32).reshape(4, 3) for k, v in raw.items()}


BOARD_PATH = ROOT / "configs" / "board.local.json"


def load_board(path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    data = json.loads(Path(path).read_text())
    return _board(data["plate"]), _board(data["base"])
