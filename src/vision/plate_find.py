"""Keeps the plate's four corners on the bright board in every frame, so the map follows a
shaking or bumped camera and the plate's own tilt. Works in the calibration's mm: the lens is
gone there and the board's edges are straight. Corners run counter-clockwise from (+x, +y)."""

from typing import NamedTuple

import cv2
import numpy as np

from src.vision.calib import plate_mm_to_px, px_to_plate_mm
from src.vision.px_to_mm import plate_px

SQUARE = np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)], np.float64)


class Grid(NamedTuple):
    """Where to look: brightness profiles across each edge of corners, inside to outside."""

    corners: np.ndarray
    x_px: np.ndarray  # (4 * samples, steps)
    y_px: np.ndarray
    along: np.ndarray  # (4, samples) mm from each edge's middle
    across: np.ndarray  # (steps,) mm out from the edge


def configured(half_mm: float, cfg: dict) -> np.ndarray:
    """Corners where the settings' centre and yaw put the plate, in the calibration's mm."""
    at = [plate_px(half_mm * x, half_mm * y, cfg) for x, y in SQUARE]
    return px_to_plate_mm(at, cfg["calib"])


def edges(corners: np.ndarray):
    """Per edge, corner to corner: its middle, unit direction, outward normal and length."""
    d = np.roll(corners, -1, axis=0) - corners
    length = np.linalg.norm(d, axis=1)
    t = d / length[:, None]
    return corners + d / 2, t, np.column_stack([t[:, 1], -t[:, 0]]), length


def grid(corners: np.ndarray, cfg: dict) -> Grid:
    """Profiles reach plate_search_mm either side of each edge and stay that far off the corners."""
    reach = cfg["plate_search_mm"]
    mid, t, n, length = edges(corners)
    across = np.arange(-reach, reach, 1 / cfg["px_per_mm"])
    along = np.linspace(-0.5, 0.5, cfg["plate_edge_samples"]) * (length[:, None] - 2 * reach)
    mm = mid[:, None, None] + along[:, :, None, None] * t[:, None, None]
    mm = mm + across[:, None] * n[:, None, None]
    px = plate_mm_to_px(mm.reshape(-1, 2), cfg["calib"]).astype(np.float32)
    x, y = (np.ascontiguousarray(v).reshape(-1, len(across)) for v in px.T)
    return Grid(corners, x, y, along, across)


def scan(image: np.ndarray, g: Grid) -> np.ndarray:
    """The picture along the grid's profiles, (4, samples, steps)."""
    # past the frame the last pixel repeats, so the frame's own border is never a step
    cut = cv2.remap(image, g.x_px, g.y_px, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if cut.ndim == 3:
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
    return cut.astype(np.float32).reshape(4, -1, len(g.across))


def steps(cut: np.ndarray, across: np.ndarray, cfg: dict) -> np.ndarray:
    """Mm across the edge where each profile first drops from bright to darker, nan if it never
    does. The first drop, not the biggest: past the board there can be a dimmer strip."""
    w = max(1, round(cfg["plate_edge_width_mm"] * cfg["px_per_mm"]))
    total = np.cumsum(np.pad(cut, ((0, 0), (0, 0), (1, 0))), axis=-1)
    inner = (total[..., w:-w] - total[..., : -2 * w]) / w
    drop = inner - (total[..., 2 * w :] - total[..., w:-w]) / w
    hit = drop >= cfg["plate_edge_step"] * inner
    first, last = hit.argmax(-1), drop.shape[-1] - 1
    # the steepest point of that first drop, then a parabola through it for the fraction
    lobe = np.minimum(first[..., None] + np.arange(2 * w), last)
    peak = first + np.take_along_axis(drop, lobe, -1).argmax(-1)
    near = np.clip(peak[..., None] + (-1, 0, 1), 0, last)
    a, b, c = np.moveaxis(np.take_along_axis(drop, near, -1), -1, 0)
    frac = np.clip(0.5 * (a - c) / np.minimum(a - 2 * b + c, -1e-9), -0.5, 0.5)
    mm = across[0] + (peak + frac + w - 0.5) * (across[1] - across[0])
    return np.where(hit.any(-1), mm, np.nan)


def fit_edges(along: np.ndarray, off: np.ndarray, cfg: dict):
    """Per edge the line most of its steps agree on, as mm out and slope at the edge's middle,
    and whether enough agree. A hand, a glint or the ball leaves its steps off the line."""
    i, j = np.triu_indices(along.shape[1], 1)
    slope = (off[:, j] - off[:, i]) / (along[:, j] - along[:, i])
    at = off[:, i, None] + slope[..., None] * (along[:, None] - along[:, i, None])
    # a missing step is nan and never near
    near = np.abs(at - off[:, None]) <= cfg["plate_edge_inlier_mm"]
    best = near[np.arange(len(off)), near.sum(-1).argmax(-1)]
    n = best.sum(-1)
    seen = n >= max(2, cfg["plate_edge_min_frac"] * along.shape[1])
    # least squares through the agreeing steps
    w = best / np.maximum(n, 1)[:, None]
    o = np.nan_to_num(off)
    s0, o0 = (w * along).sum(-1), (w * o).sum(-1)
    ds, do = along - s0[:, None], o - o0[:, None]
    slope = (w * ds * do).sum(-1) / np.maximum((w * ds * ds).sum(-1), 1e-9)
    return o0 - slope * s0, slope, seen


def edge_lines(corners: np.ndarray, out=0.0, slope=0.0) -> np.ndarray:
    """The four edges as homogeneous lines, moved out and turned by a fit when given."""
    mid, t, n, _ = edges(corners)
    p, d = mid + np.c_[out] * n, t + np.c_[slope] * n
    one = np.ones((4, 1))
    return np.cross(np.hstack([p, one]), np.hstack([p + d, one]))


def crossings(lines: np.ndarray) -> np.ndarray:
    """Corners where each edge line meets the one before it."""
    at = np.cross(np.roll(lines, 1, axis=0), lines)
    return at[:, :2] / at[:, 2:]


def board_like(corners: np.ndarray, half_mm: float, cfg: dict) -> bool:
    """The corners are the picture of a real square of that size: seen through the lens its two
    sides come out equally long, square to each other, and as far away as the calibration sheet."""
    back = np.linalg.inv(cfg["calib"]["H"])
    rays = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), back).astype(np.float32)
    g = cv2.getPerspectiveTransform((half_mm * SQUARE).astype(np.float32), rays)
    a, b = np.linalg.norm(g[:, :2], axis=0)
    sheet = np.linalg.norm(back[:, 0] / back[2, 2])
    off = [a / b - 1, g[:, 0] @ g[:, 1] / (a * b), (a + b) / (2 * sheet) - 1]
    return bool(np.all(np.abs(off) <= cfg["plate_shape_tol"]))


class PlateTracker:
    """Starts from the corners it is handed. lost while fewer than two edges show or the edges
    do not make a plate-sized square; the corners then stay where they were."""

    def __init__(self, cfg: dict, half_mm: float, corners: np.ndarray):
        self.cfg, self.half_mm, self.corners = cfg, half_mm, corners
        self.grid = grid(corners, cfg)
        self.lost = True

    def update(self, image: np.ndarray) -> None:
        cfg, g = self.cfg, self.grid
        out, slope, seen = fit_edges(g.along, steps(scan(image, g), g.across, cfg), cfg)
        lines = edge_lines(g.corners, out, slope)
        if not seen.all():
            # an edge with no fit keeps the line it had
            lines = np.where(seen[:, None], lines, edge_lines(self.corners))
        new = crossings(lines)
        self.lost = int(seen.sum()) < 2 or not board_like(new, self.half_mm, cfg)
        if self.lost:
            return
        self.corners = new
        if np.abs(new - g.corners).max() > cfg["plate_search_mm"] / 2:
            # the board has moved off the middle of the search, so look around where it is now
            self.grid = grid(new, cfg)

    def mapped(self) -> dict:
        """The settings with the map moved onto the corners: calibration mm straight to plate mm."""
        square = (self.half_mm * SQUARE).astype(np.float32)
        to_plate = cv2.getPerspectiveTransform(self.corners.astype(np.float32), square)
        calib = {**self.cfg["calib"], "H": to_plate @ np.array(self.cfg["calib"]["H"])}
        return {**self.cfg, "calib": calib, "calib_centre_mm": [0.0, 0.0], "calib_yaw_deg": 0.0}
