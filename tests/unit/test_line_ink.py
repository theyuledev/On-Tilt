import numpy as np
import pytest

from tests.unit.line_pictures import S_CURVE, VIS, follows, picture, run

pytestmark = pytest.mark.unit

WIPED = [(-60.0, -80.0), (60.0, -100.0)]


def test_what_a_wipe_left_beside_a_dark_stroke_is_not_ink():
    # the stroke is 17% darker than the board, the wiped line 6%: over line_core, so ink alone
    wiped = picture(VIS, [WIPED], 4, shade=193)
    assert run(wiped).reason == ""
    follows(run(np.minimum(picture(VIS, [S_CURVE], 4, shade=170), wiped)), S_CURVE)
