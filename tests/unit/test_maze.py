import pytest

from src.games import maze

pytestmark = pytest.mark.unit


def test_square_loads_as_float_tuples():
    m = maze.load("square")
    assert m.points == (
        (85.0, 85.0),
        (-85.0, 85.0),
        (-85.0, -85.0),
        (85.0, -85.0),
        (85.0, 85.0),
    )
    assert m.checkpoints == (1, 2, 3, 4)
    assert m.start == (85.0, 85.0)
    assert m.walls == ()
    assert all(type(v) is float for p in m.points for v in p)


def write(tmp_path, body):
    (tmp_path / "m.toml").write_text("points = [[0, 0], [10, 5.5]]\nstart = [0, 0]\n" + body)
    return tmp_path


def test_walls_load(tmp_path):
    walls = "walls = [[[0, 0], [5, 0]], [[1, 2], [3, 4], [5, 6]]]\n"
    m = maze.load("m", write(tmp_path, "checkpoints = [1]\n" + walls))
    assert m.walls == (((0.0, 0.0), (5.0, 0.0)), ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
    assert m.points[1] == (10.0, 5.5)


@pytest.mark.parametrize(
    "body",
    ["checkpoints = []\n", "checkpoints = [2]\n", "checkpoints = [-1]\n"],
)
def test_bad_checkpoints_raise(tmp_path, body):
    with pytest.raises(ValueError, match="checkpoint"):
        maze.load("m", write(tmp_path, body))


def test_bad_start_raises(tmp_path):
    (tmp_path / "m.toml").write_text("points = [[0, 0]]\ncheckpoints = [0]\nstart = [0]\n")
    with pytest.raises(ValueError, match="start"):
        maze.load("m", tmp_path)
