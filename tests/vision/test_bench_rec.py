import json
from collections import Counter
from pathlib import Path

import pytest

from src.io.settings import load
from src.vision.bench import rec
from src.vision.pipeline import Pipeline
from tests.unit.test_plate_find import CALIB
from tools import replay_clip

pytestmark = pytest.mark.unit

GAP_S = 2.0


def snap(t_s: float, **more) -> dict:
    return {"t_s": t_s, "armed": True, "ball": {"valid": True}, "plate_lost": False, **more}


def folder(tmp_path: Path, lines: list, frames: list[float], local: str = "") -> Path:
    """A recorder folder: lines are snapshots, or raw text for a line that will not parse."""
    out = tmp_path / "2026-01-01_0000_drawn"
    (out / "frames").mkdir(parents=True)
    text = [x if isinstance(x, str) else json.dumps(x) for x in lines]
    (out / "snap.jsonl").write_text("".join(line + "\n" for line in text))
    for t_s in frames:
        (out / "frames" / f"{t_s:010.3f}.jpg").touch()
    (out / "settings.local.toml").write_text(local)
    return out


def test_each_frame_is_paired_with_the_snapshot_of_its_stamp(tmp_path):
    rows = [snap(100.0), snap(100.05), snap(100.1), snap(100.15)]
    out = folder(tmp_path, rows, [100.05])
    got = rec.read(out, GAP_S)
    assert got.shots == [rec.Shot(out / "frames" / "000100.050.jpg", rows[1], rows[2], 0)]
    assert got.frames == 1 and not got.skipped


def test_the_folders_own_settings_win_over_the_base_file(tmp_path):
    base = load()["vision"]
    out = folder(tmp_path, [], [], f"[vision]\nmin_fill = {{ value = {base['min_fill'] / 2} }}\n")
    got = rec.settings(out)["vision"]
    assert got["min_fill"] == base["min_fill"] / 2
    assert got["min_ball_r_px"] == base["min_ball_r_px"]


def test_a_folder_without_its_settings_is_refused(tmp_path):
    out = folder(tmp_path, [], [])
    (out / "settings.local.toml").unlink()
    with pytest.raises(FileNotFoundError, match="not a recorder folder"):
        rec.settings(out)


def test_rows_without_plate_lost_or_outline_still_read(tmp_path):
    old = {"t_s": 100.0, "armed": False, "ball": {"valid": False}}
    lost = snap(100.05, plate_lost=True, plate_outline_px=[[1.0, 2.0]])
    got = rec.read(folder(tmp_path, [old, lost, snap(100.1)], [100.0]), GAP_S)
    assert [s.snap for s in got.shots] == [old]
    assert got.counts["board not recorded"] == 1 and got.counts["board lost"] == 1
    assert got.counts["armed"] == 2 and got.counts["seen"] == 2


def test_a_gap_or_restart_starts_a_new_stretch(tmp_path):
    late = 100.05 + GAP_S + 0.01
    rows = [snap(100.0), snap(100.05), snap(late), snap(103.0), snap(50.0), snap(50.05)]
    got = rec.read(folder(tmp_path, rows, [100.0, late, 50.0]), GAP_S)
    assert [s.stretch for s in got.shots] == [0, 1, 2]
    assert got.stretches == 3


def test_skipped_frames_are_counted_with_a_reason(tmp_path):
    rows = [snap(100.0), snap(100.05), snap(200.0), "{not json", snap(200.05), snap(200.1)]
    out = folder(tmp_path, rows, [100.0, 100.05, 200.1, 999.0])
    got = rec.read(out, GAP_S)
    assert [s.snap["t_s"] for s in got.shots] == [100.0]
    # 100.05 is followed by a gap, 200.1 by nothing, 999 has no snapshot at all
    assert got.skipped == Counter({rec.NO_NEXT: 2, rec.NO_STAMP: 1})
    assert got.counts["torn"] == 1 and got.counts["snapshots"] == 5
    text = rec.describe(out.name, got)
    assert f"skipped 3 of 4 frames: 2 {rec.NO_NEXT}, 1 {rec.NO_STAMP}" in text
    assert "1 torn lines" in text


def test_the_catalogue_calibration_reaches_the_pipeline_without_the_local_file(tmp_path):
    out = folder(tmp_path, [], [], "[vision]\nuse_calib = true\nplate_track = true\n")
    calib = tmp_path / "calib_of_that_day.json"
    calib.write_text(json.dumps(CALIB))
    cfg = rec.settings(out)
    with pytest.raises(FileNotFoundError):
        Pipeline(cfg)  # this checkout has no calibration of its own
    pipeline = Pipeline(rec.calibrated(cfg, calib))
    assert pipeline.cfg["calib"] == CALIB and pipeline.plate is not None
    flat = {**cfg, "vision": {**cfg["vision"], "use_calib": False}}
    assert rec.calibrated(flat, tmp_path / "missing.json") is flat


def test_a_folder_argument_prints_the_inventory_with_its_skipped_line(tmp_path, capsys):
    out = folder(tmp_path, [snap(100.0), snap(100.05), snap(100.1)], [100.0])
    replay_clip.main([str(out)])
    text = capsys.readouterr().out
    assert text.startswith(f"{out.name}: 1 frames, 3 snapshots, 0 torn lines, 1 stretches")
    assert "1 frames paired, snapshot to next median 50 ms, longest 50 ms" in text
    assert text.rstrip().endswith("skipped 0 of 1 frames")
