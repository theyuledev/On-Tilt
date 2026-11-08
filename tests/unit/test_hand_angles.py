import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.inputs.hand import HandIntent, palm_angles, palm_center
from src.inputs.hand_process import reading_for
from tests.unit.test_hand import CFG

pytestmark = pytest.mark.unit

TILT = {**CFG, "hand_mode": "tilt"}
# a right hand flat under the camera in metres, fingers pointing away from the player
# image axes: x right, y down (towards the player), z away from the camera
FLAT = np.zeros((21, 3))
FLAT[5] = (0.03, -0.08, 0.0)
FLAT[9] = (0.01, -0.09, 0.0)
FLAT[17] = (-0.03, -0.07, 0.0)


def rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def world(pts):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in pts]


def mirror(pts):
    return pts * (-1, 1, 1)


def test_a_flat_palm_reads_zero():
    assert palm_angles(world(FLAT)) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_turning_about_y_is_roll_and_about_x_is_pitch():
    roll, pitch = palm_angles(world(FLAT @ rot_y(20).T))
    assert abs(roll) == pytest.approx(20) and pitch == pytest.approx(0, abs=1e-6)
    roll, pitch = palm_angles(world(FLAT @ rot_x(20).T))
    assert abs(pitch) == pytest.approx(20) and roll == pytest.approx(0, abs=1e-6)


def test_opposite_tilts_read_opposite_signs():
    assert palm_angles(world(FLAT @ rot_y(-20).T))[0] == pytest.approx(
        -palm_angles(world(FLAT @ rot_y(20).T))[0]
    )
    assert palm_angles(world(FLAT @ rot_x(-20).T))[1] == pytest.approx(
        -palm_angles(world(FLAT @ rot_x(20).T))[1]
    )


def test_the_other_hand_at_the_same_tilt_reads_the_same():
    assert palm_angles(world(mirror(FLAT))) == pytest.approx((0.0, 0.0), abs=1e-6)
    pitched = FLAT @ rot_x(20).T
    assert palm_angles(world(mirror(pitched))) == pytest.approx(palm_angles(world(pitched)))
    tilt = (rot_y(15) @ rot_x(-25)).T
    assert palm_angles(world(mirror(FLAT) @ tilt)) == pytest.approx(palm_angles(world(FLAT @ tilt)))


def right_side_down(pts, deg):
    return pts @ rot_y(-deg).T


def fingertips_down(pts, deg):
    return pts @ rot_x(-deg).T


def steer(pts, cfg=TILT):
    hand = HandIntent(cfg)
    hand.update((0.0, 0.0), 0.0, recentre=True)
    return hand.update(palm_angles(world(pts)), 0.1)


@pytest.mark.parametrize("flat", [FLAT, mirror(FLAT)], ids=["right hand", "left hand"])
def test_right_side_down_steers_plus_x_and_fingertips_down_plus_y(flat):
    assert palm_angles(world(right_side_down(flat, 20))) == pytest.approx((20, 0), abs=1e-6)
    i = steer(right_side_down(flat, 20))
    assert i.x > 0 and i.y == pytest.approx(0, abs=1e-6)
    assert palm_angles(world(fingertips_down(flat, 20))) == pytest.approx((0, -20), abs=1e-6)
    i = steer(fingertips_down(flat, 20))
    assert i.y > 0 and i.x == pytest.approx(0, abs=1e-6)


def test_flips_negate_their_own_axis():
    pts = fingertips_down(right_side_down(FLAT, 15), 10)
    i = steer(pts)
    fx = steer(pts, {**TILT, "hand_flip_x": True})
    fy = steer(pts, {**TILT, "hand_flip_y": True})
    assert (fx.x, fx.y) == (-i.x, i.y) and (fy.x, fy.y) == (i.x, -i.y)


def test_a_palm_with_no_normal_reads_as_no_hand():
    assert palm_angles(world(np.zeros((21, 3)))) is None


def test_hand_mode_picks_the_reading_and_a_bad_one_stops_at_startup():
    spots = [SimpleNamespace(x=0.1 * i, y=0.5, z=0.0) for i in range(21)]
    result = SimpleNamespace(
        hand_landmarks=[spots], hand_world_landmarks=[world(right_side_down(FLAT, 20))]
    )
    assert reading_for("tilt")(result) == pytest.approx((20, 0), abs=1e-6)
    assert reading_for("palm")(result) == palm_center(spots)
    with pytest.raises(SystemExit, match="hand_mode"):
        reading_for("roll")
