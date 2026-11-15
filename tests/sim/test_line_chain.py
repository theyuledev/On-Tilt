import math

import path_rig as rig
import pytest

from tests.unit.line_pictures import S_CURVE, VIS, ball_px, picture
from tests.unit.test_web_line import Board, read

pytestmark = pytest.mark.sim

BALL_MM = 8.0


def test_a_drawn_s_line_read_with_one_click_is_followed_from_the_ball_to_its_other_end():
    start, end = S_CURVE[-1], S_CURVE[0]
    cfg = rig.rig_cfg(rig.STRAIGHT)
    cfg["control"]["mode"] = "balance"
    loop = Board([picture(VIS, [S_CURVE], 4, ball=(start, BALL_MM))], cfg=cfg)
    loop.ball_px = ball_px(VIS, start, BALL_MM)
    assert read(loop)["line"] == "ok" and loop.mode == "path"
    loop.arm()
    assert loop.safety.armed
    # the awkward sim plate: seen 0.12 s late, two blind stretches, a ball that sticks
    r = rig.follow(S_CURVE, start=start, end=end, cfg=loop.cfg, mode=loop.step)
    followed = loop.step.path_view()
    assert math.dist(followed["points"][0], start) < 10
    assert math.dist(followed["points"][-1], end) < 10
    # measured: at the end in 5.7 s, done at 5.9 s, 9.8 mm off the drawn line at worst, 4.7 median
    assert r["finished"] and r["time_s"] < 10.0 and followed["done"]
    assert abs(r["done_s"] - r["time_s"]) < 0.5
    assert r["off_max_mm"] < 12.0 and r["off_median_mm"] < 6.0 and r["end_mm"] < 15.0
