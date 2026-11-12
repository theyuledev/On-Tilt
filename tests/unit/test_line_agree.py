import time

import numpy as np
import pytest

from src.vision.line_path import DIFFER, agree
from tests.unit.line_pictures import STRAIGHT, VIS, follows, picture, run

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["black", "white", "noise"])
def test_a_blank_or_noise_frame_is_refused_fast(kind):
    img = {
        "black": np.zeros((800, 1280), np.uint8),
        "white": np.full((800, 1280), 255, np.uint8),
        "noise": np.random.default_rng(0).integers(0, 256, (800, 1280), dtype=np.uint8),
    }[kind]
    start = time.perf_counter()
    assert run(img).reason == "no line found"
    assert time.perf_counter() - start < 1


def test_five_reads_give_the_path_most_agree_on():
    full, short = picture(VIS, [STRAIGHT]), picture(VIS, [[STRAIGHT[0], (0.0, 10.0)]])
    reads = [run(full)] * 3 + [run(short), run(picture(VIS, []))]
    assert reads[3].reason == "" and reads[4].reason == "no line found"
    follows(agree(reads, VIS), STRAIGHT)


def test_reads_that_do_not_agree_are_refused():
    other = [(-90.0, 60.0), (80.0, 60.0)]
    reads = [run(picture(VIS, [STRAIGHT]))] * 2 + [run(picture(VIS, [other]))] * 2
    assert agree(reads + [run(picture(VIS, []))], VIS) == ([], DIFFER)
    blank = run(picture(VIS, []))
    assert agree([blank] * 3 + reads[:2], VIS) == ([], "no line found")
