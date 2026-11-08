import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.inputs.hand import palm_angles

pytestmark = pytest.mark.unit

# a right hand flat to the camera in metres, image axes: x right, y down, z away
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
