import math

import cv2
import pytest

from src.vision.markers import TiltTracker
from src.vision.plate_pose import tilt_report
from tests.unit.test_plate_pose import STANDARD, T, pose
from tests.vision.test_markers import BASE, CFG, PLATE, R_BASE, T_BASE, render
from tools.servo_test import annotate, load_csv, save

pytestmark = pytest.mark.unit
CFG2 = {**CFG, "marker_min_count": 2}
C, US_PER_DEG, LEVEL, TICK = 1500, 28.75, 0.8, 0.02
HIDDEN = range(23, 30)
FRAMES: dict = {}


def steps() -> list[tuple[int, str]]:
    out = [(C, "find")] * 5 + [(C, "hold")] * 30
    ramp = [*range(C + 25, C + 225, 25), *range(C + 175, C - 225, -25), *range(C - 175, C + 25, 25)]
    out += [(us, "sweep") for us in ramp]
    for d in (0, 100, 0, -100, 0):
        out += [(C + d, "step")] * 15
    return out


def frame(us: int | None):
    """The plate at LEVEL plus the pulse over US_PER_DEG, or only the base when us is None."""
    if us not in FRAMES:
        boards = [(BASE, R_BASE, T_BASE)]
        if us is not None:
            boards.append((PLATE, pose(tip_y=LEVEL + (us - C) / US_PER_DEG)[0], T))
        FRAMES[us] = render(boards)
    return FRAMES[us]


def plate_run():
    tracker = TiltTracker(CFG2, STANDARD, PLATE, BASE)
    commands, rows = [], []
    for i, (us, phase) in enumerate(steps()):
        t = i * TICK
        commands.append((t, t + 0.001, us, 1500, phase))
        rows.append(tracker.update(frame(None if i in HIDDEN else us), t + 0.002, us))
    return commands, rows


def test_marker_run_gives_the_slope_sign_and_hold_level(tmp_path, monkeypatch):
    import tools.servo_test as st

    monkeypatch.setattr(st, "OUT", tmp_path)
    commands, rows = plate_run()
    hidden = [r for r in rows if math.isnan(r[1])]
    assert len(hidden) == len(HIDDEN) and all(r[3] == 0 for r in hidden)
    out = tilt_report(*load_csv(save("p", commands, rows, "tilt"))[:2], 0)
    assert out["frames"] == len(rows) - len(HIDDEN)
    assert out["deg_per_us_sweep"] == pytest.approx(1 / US_PER_DEG, rel=0.03)
    assert out["deg_per_us"] == pytest.approx(1 / US_PER_DEG, rel=0.03)
    # the hidden stretch sits in the half of the hold the level is read from
    assert out["hold_deg"] == pytest.approx(LEVEL, abs=0.2)
    assert out["trim_us"] == pytest.approx(-LEVEL * US_PER_DEG, abs=6)


def test_annotate_takes_the_bgr_frame(tmp_path):
    tracker = TiltTracker(CFG2, STANDARD, PLATE, BASE)
    bgr = cv2.cvtColor(frame(C), cv2.COLOR_GRAY2BGR)
    assert not math.isnan(tracker.update(bgr, 0.0)[1])
    annotate(tracker, tmp_path / "p.jpg")
    assert cv2.imread(str(tmp_path / "p.jpg")).shape == bgr.shape


def test_finish_on_a_run_that_never_saw_the_plate_says_so(tmp_path, monkeypatch, capsys):
    import tools.servo_test as st

    monkeypatch.setattr(st, "OUT", tmp_path)
    tracker = TiltTracker(CFG2, STANDARD, PLATE, BASE)
    rows = [tracker.update(frame(None), i * TICK) for i in range(12)]
    commands = [(i * TICK, i * TICK, C, 1500, "hold") for i in range(12)]
    st.finish("blind", commands, rows, tracker, "tilt", {}, False)
    assert "target not found" in capsys.readouterr().out
    assert (tmp_path / "blind.csv").exists() and not (tmp_path / "blind.jpg").exists()
