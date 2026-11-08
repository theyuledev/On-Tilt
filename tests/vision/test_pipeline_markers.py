import math

import cv2
import numpy as np
import pytest

from src.io.settings import load
from src.vision import pipeline
from src.vision.calib import undistort
from src.vision.markers import relative_tilt
from src.vision.pipeline import Pipeline
from src.vision.plate_frame import plate_to_px
from src.vision.px_to_mm import ball_state, to_mm
from src.vision.tracker import BallTracker
from tests.unit.test_plate_pose import STANDARD, T, pose, rot
from tests.vision.test_markers import BASE, PLATE, R_BASE, T_BASE, render, truth

pytestmark = pytest.mark.unit

CFG = load()
R_BALL = CFG["rig"]["ball_d_mm"] / 2
BALL_MM = (40.0, -20.0)
SHADE = 25
FLOOR = 40
HALF = CFG["rig"]["plate_inner_mm"] / 2


def pipe(boards=(PLATE, BASE)):
    return Pipeline(CFG, lens=STANDARD, boards=boards)


def with_ball(frame, r, t, xy=BALL_MM):
    """Dark disc where the ball's centre projects, its radius from a point one radius over."""
    rvec = cv2.Rodrigues(r)[0]
    c, edge = plate_to_px([xy, (xy[0] + R_BALL, xy[1])], STANDARD, rvec, t, z_mm=R_BALL)
    r_px = float(np.linalg.norm(edge - c))
    cv2.circle(
        frame, tuple((c * 16).round().astype(int)), int(r_px * 16), SHADE, -1, cv2.LINE_AA, 4
    )
    return frame


def erase(frame, board, r, t):
    for pts in board.values():
        quad = truth(pts.mean(0) + (pts - pts.mean(0)) * 1.6, r, t)
        cv2.fillConvexPoly(frame, quad.round().astype(np.int32), 200)
    return frame


def scene(tip_y=4.0, tip_x=-3.0):
    r, tx, ty = pose(tip_y, tip_x)
    frame = with_ball(render([(PLATE, r, T), (BASE, R_BASE, T_BASE)]), r, T)
    return frame, r, (tx, ty)


def test_tilt_ball_and_corners_from_a_marker_frame():
    frame, r, tilt = scene()
    o = pipe().step(frame, 0.0)
    assert not o.occluded
    # the two base markers put a steady ~0.3 deg on tilt_y, so the base is held looser
    assert (o.tilt_x_deg, o.tilt_y_deg) == pytest.approx(tilt, abs=0.5)
    assert o.ball.valid and (o.ball.x_mm, o.ball.y_mm) == pytest.approx(BALL_MM, abs=1.0)
    alone = pipe().step(with_ball(render([(PLATE, r, T)]), r, T), 0.0)
    assert (alone.tilt_x_deg, alone.tilt_y_deg) == pytest.approx(tilt, abs=0.2)
    assert (alone.ball.x_mm, alone.ball.y_mm) == pytest.approx(BALL_MM, abs=1.0)
    half = CFG["rig"]["plate_inner_mm"] / 2
    corners = [[-half, half], [half, half], [half, -half], [-half, -half]]
    want = plate_to_px(corners, STANDARD, cv2.Rodrigues(r)[0], T)
    assert np.abs(o.plate_corners_px - want).max() < 0.5


def test_plate_markers_hidden_gives_nan_and_occluded_but_keeps_the_ball():
    frame, r, _ = scene()
    o = pipe().step(erase(frame, PLATE, r, T), 0.0)
    assert o.occluded and math.isnan(o.tilt_x_deg) and math.isnan(o.tilt_y_deg)
    assert o.plate_corners_px is None
    assert o.ball.valid and o.ball_px.r_px > 30
    vis = CFG["vision"]
    cx, cy = vis["plate_centre_px"]
    flat = ((o.ball_px.x_px - cx) / vis["px_per_mm"], (cy - o.ball_px.y_px) / vis["px_per_mm"])
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx(flat)


def test_base_seen_once_is_remembered_when_hidden():
    # a 2 deg nod of the whole stand: the base cancels it, the raw camera tilt does not
    r, tx, ty = pose(tip_y=3, tip_x=-2)
    s = rot([1, 0, 0], 2)
    both = render([(PLATE, s @ r, s @ T), (BASE, s @ R_BASE, s @ T_BASE)], seed=1)
    alone = render([(PLATE, s @ r, s @ T)], seed=2)
    p = pipe()
    first = p.step(both, 0.0)
    second = p.step(alone, 0.01)
    for o in (first, second):
        assert (o.tilt_x_deg, o.tilt_y_deg) == pytest.approx((tx, ty), abs=0.5)
    assert (second.tilt_x_deg, second.tilt_y_deg) == pytest.approx(first[2:4], abs=0.05)
    raw = relative_tilt(s @ r)
    assert max(abs(raw[0] - tx), abs(raw[1] - ty)) > 1.0
    never = pipe().step(alone, 0.0)
    assert (never.tilt_x_deg, never.tilt_y_deg) == pytest.approx(raw, abs=0.3)


def test_board_ids_off_the_settings_print_once(capsys):
    pipe()
    assert capsys.readouterr().out == ""
    pipe((PLATE, {4: BASE[4], 6: BASE[5]}))
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and "base" in out[0] and "[4, 6]" in out[0]


def test_hidden_markers_keep_the_last_floor_white():
    frame, r, _ = scene()
    p = pipe()
    seen = []
    update = p.tracker.update
    p.tracker.update = lambda img, t: seen.append(img) or update(img, t)
    p.step(frame, 0.0)
    o = p.step(erase(frame.copy(), PLATE, r, T), 0.01)
    painted = (seen[0] == 255) & (frame < 255)
    assert o.occluded and painted.mean() > 0.3 and (seen[1][painted] == 255).all()

def floor_past_left_edge(frame, r, t, width_mm=60.0):
    """Dark floor from the plate's -x edge out to width_mm, through the lens like the prints."""
    ys, xs = np.mgrid[: frame.shape[0], : frame.shape[1]]
    rays = undistort(STANDARD, np.c_[xs.ravel(), ys.ravel()]).reshape(-1, 1, 2)
    uv = cv2.perspectiveTransform(rays, np.linalg.inv(np.c_[r[:, :2], t])).reshape(xs.shape + (2,))
    out = -uv[..., 0] - HALF
    frame[(out > 0) & (out < width_mm)] = FLOOR
    return frame


def test_ball_against_the_edge_stays_found_with_floor_past_it():
    r, _, _ = pose(4.0, -3.0)
    xy = (-HALF + R_BALL, 0.0)
    frame = render([(PLATE, r, T), (BASE, R_BASE, T_BASE)])
    frame = with_ball(floor_past_left_edge(frame, r, T), r, T, xy)
    assert not BallTracker(CFG["vision"]).update(frame, 0.0).valid
    o = pipe().step(frame, 0.0)
    assert o.ball.valid and (o.ball.x_mm, o.ball.y_mm) == pytest.approx(xy, abs=1.5)


@pytest.mark.parametrize("lens, boards", [(None, (PLATE, BASE)), (STANDARD, None)])
def test_no_lens_or_no_board_file_is_the_ball_only_path(monkeypatch, tmp_path, lens, boards):
    monkeypatch.setattr(pipeline, "CALIB_PATH", tmp_path / "calib.json")
    monkeypatch.setattr(pipeline, "BOARD_PATH", tmp_path / "board.json")
    frame, _, _ = scene()
    o = Pipeline(CFG, lens=lens, boards=boards).step(frame, 0.0)
    px = BallTracker(CFG["vision"]).update(frame, 0.0)
    assert o.ball_px == px and o.ball == ball_state(px, [], CFG["vision"])
    assert math.isnan(o.tilt_x_deg) and math.isnan(o.tilt_y_deg)
    assert not o.occluded and o.plate_corners_px is None


def _throw(*_):
    raise cv2.error("degenerate")


@pytest.mark.parametrize(
    "name, fake",
    [
        ("board_pose", _throw),
        ("board_pose", lambda *_: (np.full(3, np.nan), np.full(3, np.nan))),
        ("ball_on_plate", lambda *_: (math.inf, math.nan)),
    ],
)
def test_a_degenerate_solve_falls_back_to_the_flat_route(monkeypatch, name, fake):
    monkeypatch.setattr(pipeline, name, fake)
    frame, _, _ = scene()
    o = pipe().step(frame, 0.0)
    assert o.occluded and math.isnan(o.tilt_x_deg) and o.plate_corners_px is None
    assert o.ball.valid
    assert (o.ball.x_mm, o.ball.y_mm) == pytest.approx(to_mm(o.ball_px, CFG["vision"]))
