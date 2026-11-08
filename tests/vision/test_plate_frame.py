import cv2
import numpy as np
import pytest

from src.vision.calib import project
from src.vision.plate_frame import ball_on_plate, plate_to_px
from tests.unit.test_plate_pose import FISHEYE, STANDARD, T, pose, rot

pytestmark = pytest.mark.unit
LENSES = pytest.mark.parametrize("lens", [STANDARD, FISHEYE], ids=lambda c: c["model"])
LACROSSE, SMALL = 30.9, 6.5


@LENSES
def test_ball_round_trips_under_random_poses(lens):
    rng = np.random.default_rng(0)
    for i in range(50):
        tip_x, tip_y = rng.uniform(-8, 8, 2)
        # yaw on the plate side so plate axes are not image aligned
        r = pose(tip_y, tip_x)[0] @ rot([0, 0, 1], rng.uniform(-180, 180))
        rvec = cv2.Rodrigues(r)[0]
        x, y = rng.uniform(-140, 140, 2)
        r_ball = (LACROSSE, SMALL)[i % 2]
        px = project(lens, [[x, y, r_ball]], rvec, T)
        assert ball_on_plate(px[0], lens, rvec, T, r_ball) == pytest.approx((x, y), abs=0.01)


@LENSES
def test_plate_to_px_inverts_and_projects_the_origin(lens):
    rvec = cv2.Rodrigues(pose(tip_y=3, tip_x=-5)[0] @ rot([0, 0, 1], 20))[0]
    xy = [[50.0, -70.0], [-120.0, 30.0]]
    px = plate_to_px(xy, lens, rvec, T, z_mm=LACROSSE)
    assert px.shape == (2, 2)
    for p, want in zip(px, xy, strict=True):
        assert ball_on_plate(p, lens, rvec, T, LACROSSE) == pytest.approx(want, abs=0.01)
    level = cv2.Rodrigues(pose()[0])[0]
    assert np.allclose(plate_to_px([[0, 0]], lens, level, T), project(lens, [[0, 0, 0]], level, T))


@LENSES
def test_ray_beats_a_flat_px_per_mm_scale_at_the_edge(lens):
    r = pose()[0]
    rvec = cv2.Rodrigues(r)[0]
    px = project(lens, [[140, 0, LACROSSE]], rvec, T)[0]
    ray_err = abs(ball_on_plate(px, lens, rvec, T, LACROSSE)[0] - 140)
    # flat route: one px/mm scale from the ball's own size, offset from the plate centre px,
    # turned into plate axes with the level pose's in-plane rotation
    s = lens["K"][0][0] / (T[2] - LACROSSE)
    centre = project(lens, [[0, 0, LACROSSE]], rvec, T)[0]
    flat_err = abs((r[:2, :2].T @ ((px - centre) / s))[0] - 140)
    msg = f"ray {ray_err:.3f} mm, flat {flat_err:.3f} mm"
    assert ray_err < 0.05, msg
    if lens is STANDARD:
        assert flat_err > 3, msg


def test_tipped_plate_keeps_the_x_sign():
    r, tilt_x, _ = pose(tip_y=6)
    assert tilt_x > 0  # +x edge is the low one, farther from the cam
    rvec = cv2.Rodrigues(r)[0]
    for x in (100.0, -100.0):
        px = project(STANDARD, [[x, 0, LACROSSE]], rvec, T)[0]
        assert ball_on_plate(px, STANDARD, rvec, T, LACROSSE) == pytest.approx((x, 0), abs=0.01)
