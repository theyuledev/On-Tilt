import numpy as np
import pytest

from src.vision.maze_path import shortest_path

pytestmark = pytest.mark.unit


def test_a_fully_blocked_maze_has_no_path():
    clear = np.zeros((7, 7), np.float32)
    assert shortest_path(clear, (2, 2), (4, 4), 10.0) is None


def test_a_path_cannot_squeeze_between_touching_wall_corners():
    clear = np.zeros((7, 7), np.float32)
    clear[2, 2] = clear[3, 3] = 1.0
    assert shortest_path(clear, (2, 2), (3, 3), 10.0) is None


def test_a_path_cannot_wrap_from_one_image_edge_to_the_other():
    clear = np.zeros((4, 5), np.float32)
    clear[1, 4] = clear[2, 0] = 1.0
    assert shortest_path(clear, (4, 1), (0, 2), 10.0) is None
    assert shortest_path(clear, (0, 2), (4, 1), 10.0) is None


def test_an_open_diagonal_corridor_still_has_a_path():
    clear = np.ones((4, 4), np.float32)
    path = shortest_path(clear, (0, 0), (3, 3), 10.0)
    assert path == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_one_wall_beside_a_diagonal_forces_the_square_corner():
    clear = np.zeros((5, 5), np.float32)
    clear[1, 1] = clear[1, 2] = clear[2, 2] = 1.0
    assert shortest_path(clear, (1, 1), (2, 2), 10.0) == [(1, 1), (2, 1), (2, 2)]
