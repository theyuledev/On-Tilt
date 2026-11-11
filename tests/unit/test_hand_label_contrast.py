import numpy as np
import pytest

from src.contracts import Intent
from src.inputs.hand_view import draw

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mirror", [False, True])
def test_hand_labels_keep_a_dark_outline_on_a_bright_camera_frame(mirror):
    image = np.full((480, 640, 3), 235, np.uint8)
    output = draw(image, None, None, (12.0, -6.0), Intent(0.24, -0.12, True, 0.0), mirror)
    labels = output[:75]
    assert np.any(labels.max(axis=2) < 50)
    assert np.any(labels.min(axis=2) > 245)
