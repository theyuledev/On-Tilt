from fnmatch import fnmatch

from tests.conftest import collect_ignore_glob


def ignored(path):
    return any(fnmatch(path, glob) for glob in collect_ignore_glob)


def test_icloud_copies_ignored():
    assert ignored("tests/sim/test_hold 2.py")
    assert ignored("tests/test_x 3.py")


def test_real_tests_collected():
    assert not ignored("tests/sim/test_hold.py")
