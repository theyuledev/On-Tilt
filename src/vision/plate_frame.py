"""Plate mm from pixels through the solvePnP pose: rvec/tvec map plate coords to camera coords."""

import cv2
import numpy as np

from src.vision.calib import project, undistort


def ball_on_plate(centre_px, lens: dict, rvec, tvec, r_ball_mm: float) -> tuple[float, float]:
    """Plate x, y of a ball seen at centre_px; its centre sits one radius above the plate."""
    ray = np.append(undistort(lens, centre_px)[0], 1.0)
    rt = cv2.Rodrigues(np.asarray(rvec, np.float64))[0].T
    t = rt @ np.asarray(tvec, np.float64).ravel()
    s = (r_ball_mm + t[2]) / (rt @ ray)[2]
    x, y, _ = s * (rt @ ray) - t
    return float(x), float(y)


def plate_to_px(xy_mm, lens: dict, rvec, tvec, z_mm: float = 0.0) -> np.ndarray:
    xy = np.asarray(xy_mm, np.float64).reshape(-1, 2)
    return project(lens, np.column_stack([xy, np.full(len(xy), z_mm)]), rvec, tvec)
