import json

from src.vision import calib
from tools import calibrate, mark_board, servo_test


def test_lens_and_board_paths_point_at_tmp(tmp_path):
    lens = tmp_path / "calib.local.json"
    assert calib.CALIB_PATH == calibrate.CALIB_PATH == mark_board.CALIB_PATH == lens
    assert mark_board.OUT == servo_test.BOARD == tmp_path / "board.local.json"
    assert calib.load_calib.__defaults__ == calib.save_calib.__defaults__ == (lens,)


def test_load_calib_default_reads_tmp(tmp_path):
    (tmp_path / "calib.local.json").write_text(json.dumps({"fx": 1.0}))
    assert calib.load_calib() == {"fx": 1.0}
