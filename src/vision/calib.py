"""Lens + plate calibration from the printed checkerboard.

Plate frame comes from the sheet lying flat on the plate: origin at its centre, +x along
the 8-square side from the white corner square toward the black one (the detector's row
order), +y a quarter turn counterclockwise from +x as seen from the camera.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.io.settings import ROOT

CALIB_PATH = ROOT / "configs" / "calib.local.json"
FIND_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_corners(image: np.ndarray, pattern) -> np.ndarray | None:
    ok, corners = cv2.findChessboardCornersSB(_gray(image), tuple(pattern), flags=FIND_FLAGS)
    return corners.reshape(-1, 2).astype(np.float64) if ok else None


def board_points(pattern, square_mm: float) -> np.ndarray:
    nx, ny = pattern
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    return np.column_stack([i.ravel(), j.ravel(), np.zeros(nx * ny)]) * square_mm


def square_px(corners: np.ndarray, pattern) -> float:
    rows = corners.reshape(pattern[1], pattern[0], 2)
    return float(np.linalg.norm(np.diff(rows, axis=1), axis=2).mean())


def is_new_view(corners: np.ndarray, kept: list[np.ndarray], pattern) -> bool:
    """New if the board moved more than one square on average from every kept view."""
    step = square_px(corners, pattern)
    return all(np.linalg.norm(corners - k, axis=1).mean() > step for k in kept)


def _rms(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.reshape(err, (-1, 2)) ** 2, axis=1))))


def _kd(lens: dict):
    return np.array(lens["K"], np.float64), np.array(lens["D"], np.float64)


def project(lens: dict, xyz, rvec=None, tvec=None) -> np.ndarray:
    K, D = _kd(lens)
    xyz = np.asarray(xyz, np.float64).reshape(-1, 1, 3)
    rvec = np.zeros(3) if rvec is None else np.asarray(rvec, np.float64).ravel()
    tvec = np.zeros(3) if tvec is None else np.asarray(tvec, np.float64).ravel()
    if lens["model"] == "fisheye":
        px, _ = cv2.fisheye.projectPoints(xyz, rvec, tvec, K, D)
    else:
        px, _ = cv2.projectPoints(xyz, rvec, tvec, K, D)
    return px.reshape(-1, 2)


def undistort(lens: dict, px) -> np.ndarray:
    """Pixels to normalized camera coords (x/z, y/z)."""
    K, D = _kd(lens)
    px = np.asarray(px, np.float64).reshape(-1, 1, 2)
    if lens["model"] == "fisheye":
        n = cv2.fisheye.undistortPoints(px, K, D, criteria=CRITERIA)
    else:
        n = cv2.undistortPoints(px, K, D, criteria=CRITERIA)
    return n.reshape(-1, 2)


def _fit_fisheye(obj, views, size):
    flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
    _, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        [obj.reshape(1, -1, 3)] * len(views),
        [v.reshape(1, -1, 2) for v in views],
        size,
        np.eye(3),
        np.zeros(4),
        flags=flags,
        criteria=CRITERIA,
    )
    return {"model": "fisheye", "K": K.tolist(), "D": D.ravel().tolist()}, rvecs, tvecs


def _fit_standard(obj, views, size):
    _, K, D, rvecs, tvecs = cv2.calibrateCamera(
        [obj.astype(np.float32)] * len(views),
        [v.astype(np.float32) for v in views],
        size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
        criteria=CRITERIA,
    )
    return {"model": "standard", "K": K.tolist(), "D": D.ravel().tolist()}, rvecs, tvecs


def fit_lens(views: list[np.ndarray], pattern, square_mm: float, size) -> list[dict]:
    """Fits both lens models; best (lowest reprojection rms) first."""
    obj = board_points(pattern, square_mm)
    fits = []
    for fit in (_fit_fisheye, _fit_standard):
        try:
            lens, rvecs, tvecs = fit(obj, views, tuple(size))
        except cv2.error:
            continue
        err = [project(lens, obj, r, t) - v for v, r, t in zip(views, rvecs, tvecs, strict=True)]
        lens["rms_px"] = _rms(np.concatenate(err))
        fits.append(lens)
    if not fits:
        raise ValueError("neither lens model fits these views")
    return sorted(fits, key=lambda f: f["rms_px"])


def plate_grid(corners_px: np.ndarray, pattern, square_mm: float) -> np.ndarray:
    """Corner positions in plate mm, in the detector's order."""
    nx, ny = pattern
    xy = board_points(pattern, square_mm)[:, :2] - np.array([nx - 1, ny - 1]) * square_mm / 2
    a, b = corners_px[1] - corners_px[0], corners_px[nx] - corners_px[0]
    if a[0] * b[1] - a[1] * b[0] > 0:
        xy[:, 1] *= -1
    return xy


def px_to_plate_mm(points, calib: dict) -> np.ndarray:
    n = undistort(calib, points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(n, np.array(calib["H"])).reshape(-1, 2)


def plate_mm_to_px(points, calib: dict) -> np.ndarray:
    mm = np.asarray(points, np.float64).reshape(-1, 1, 2)
    n = cv2.perspectiveTransform(mm, np.linalg.inv(calib["H"])).reshape(-1, 2)
    return project(calib, np.column_stack([n, np.ones(len(n))]))


def fit_plate(lens: dict, corners_px: np.ndarray, pattern, square_mm: float) -> dict:
    """Lens + homography from undistorted pixels to plate mm, from the sheet lying flat."""
    mm = plate_grid(corners_px, pattern, square_mm)
    H, _ = cv2.findHomography(undistort(lens, corners_px), mm)
    calib = {**lens, "H": H.tolist()}
    calib["plate_rms_px"] = _rms(plate_mm_to_px(mm, calib) - corners_px)
    calib["plate_rms_mm"] = _rms(px_to_plate_mm(corners_px, calib) - mm)
    return calib


def plate_size_mm(image: np.ndarray, calib: dict) -> tuple[float, float] | None:
    """Edge lengths of the bright quad around the sheet, from the median of each side."""
    _, bright = cv2.threshold(_gray(image), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    origin = tuple(float(v) for v in plate_mm_to_px([[0, 0]], calib)[0])
    around = [k for k in contours if cv2.pointPolygonTest(k, origin, False) > 0]
    if not around:
        return None
    x, y = px_to_plate_mm(max(around, key=cv2.contourArea).reshape(-1, 2), calib).T
    width = np.median(x[x > abs(y)]) - np.median(x[-x > abs(y)])
    height = np.median(y[y > abs(x)]) - np.median(y[-y > abs(x)])
    return float(width), float(height)


def save_calib(calib: dict, path: Path = CALIB_PATH) -> None:
    Path(path).write_text(json.dumps(calib, indent=2) + "\n")


def load_calib(path: Path = CALIB_PATH) -> dict:
    return json.loads(Path(path).read_text())
