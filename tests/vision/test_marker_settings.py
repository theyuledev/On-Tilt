import pytest

from src.io.settings import load
from src.vision.markers import TiltTracker, detector

pytestmark = pytest.mark.unit

# read by markers.detector, TiltTracker and tools/mark_board.py
KEYS = [
    "marker_dict",
    "marker_refine_win_px",
    "marker_thresh_win_px",
    "marker_min_perimeter_rate",
    "marker_min_count",
    "plate_marker_ids",
    "base_marker_ids",
    "marker_size_mm",
]


def test_settings_have_every_marker_key():
    vision = load()["vision"]
    assert [k for k in KEYS if k not in vision] == []


def test_detector_and_tracker_build_from_settings():
    vision = load()["vision"]
    detector(vision)
    assert TiltTracker(vision, {}, {}, {}).min_count == vision["marker_min_count"]
