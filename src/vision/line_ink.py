"""Which pixels of a board picture are marker ink."""

import cv2
import numpy as np

from src.vision.balls import downsample, plate_level


def _kept(dark: np.ndarray, weak: float, core: float) -> np.ndarray:
    """Marks at least weak dark, where part of each is at least core dark."""
    _, lab = cv2.connectedComponents(np.uint8(dark > weak))
    return np.uint8(np.isin(lab, np.unique(lab[dark > core])))


def marks(dark: np.ndarray, cfg: dict, region: np.ndarray) -> np.ndarray:
    """Marks at least line_weak darker than the board around them, kept where part of them is
    line_core darker. Next to a darker stroke both limits rise: the second to line_share of the
    stroke's own darkness (the median along the centre lines of those marks inside region), the
    first to line_share of the second, so what a wipe left behind is not ink."""
    first = _kept(dark, cfg["line_weak"], cfg["line_core"])
    centre = (cv2.ximgproc.thinning(first * 255) > 0) & (region > 0)
    if not centre.any():
        return first
    stroke = float(np.median(cv2.dilate(dark, np.ones((3, 3), np.uint8))[centre]))
    core = max(cfg["line_core"], cfg["line_share"] * stroke)
    return _kept(dark, max(cfg["line_weak"], cfg["line_share"] * core), core)


def ink(gray: np.ndarray, cfg: dict, wide: int, region: np.ndarray) -> np.ndarray:
    """Thin marks darker than the board right around them. wide (px) is more than the broadest
    stroke; the stroke's darkness is measured where region is set."""
    b = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide, wide))
    board = cv2.resize(plate_level(downsample(gray, cfg), cfg, False), gray.shape[::-1])
    # capped just above the bare board so a reflection's edge can't pass for ink
    shut = np.minimum(cv2.morphologyEx(b, cv2.MORPH_CLOSE, disk), (1 + cfg["line_glare"]) * board)
    gap = round(cfg["line_gap_mm"] * cfg["px_per_mm"] / 2)
    joint = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1,) * 2)
    found = marks(1 - b / np.maximum(shut, 1.0), cfg, region)
    return cv2.morphologyEx(found, cv2.MORPH_CLOSE, joint) > 0
