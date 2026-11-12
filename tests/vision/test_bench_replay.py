import json
from collections import Counter
from pathlib import Path

import pytest

from src.vision import tracker
from src.vision.balls import Candidate
from src.vision.bench import rec, replay
from src.vision.bench.replay import MOVING, NO_TONE, WARMING
from src.vision.bench.score import ALL, CORRECT, EDGE, LOST, MIDDLE, SIDES, WRONG, Mark
from src.vision.bench.truth import ARRIVAL, UNSEEN, Label
from src.vision.tracker import BG_MIN_SAMPLES
from tests.vision.test_bench_truth_file import picture

pytestmark = pytest.mark.unit

R = 19
MID, RIGHT = (200, 150), (300, 150)
# a 120 mm board at 2 px a mm, in the middle of the 400 x 300 picture
FLAT = (
    "[rig]\nball_d_mm = { value = 16 }\nplate_inner_mm = { value = 120 }\n"
    "[vision]\nplate_centre_px = { value = [200, 150] }\npx_per_mm = { value = 2.0 }\n"
)
# pictures before a find is scored: the background samples every other one and wants five
WARM = 2 * BG_MIN_SAMPLES - 1


def name(t_s: float) -> str:
    return f"{t_s:010.3f}.jpg"


def recorded(
    tmp_path: Path, spots: list, local: str = FLAT, t0: float = 100.0, called="one"
) -> Path:
    """A recorder folder with a picture a second, the ball drawn where spots say (None: no ball).
    A new start time carries on in the same folder after a gap."""
    folder = tmp_path / called
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    (folder / "settings.local.toml").write_text(local)
    times = [t0 + i for i in range(len(spots) + 1)]
    with (folder / "snap.jsonl").open("a") as f:
        for t in times:
            f.write(json.dumps({"t_s": t, "armed": True, "ball": {"valid": True}}) + "\n")
    for t, spot in zip(times, spots, strict=False):
        picture(folder / "frames" / name(t), *([(*spot, R)] if spot else []))
    return folder


def lying(spots: list, t0: float = 100.0, at_rest: bool = True) -> dict:
    """A label on every picture with a ball, the reason a live run gives on the others."""
    return {
        name(t0 + i): Label(spot, spot, float(R), ARRIVAL, at_rest, None) if spot else UNSEEN
        for i, spot in enumerate(spots)
    }


def run(folder: Path, labels: dict) -> replay.Scored:
    cfg = rec.settings(folder)
    return replay.run(rec.read(folder, cfg["vision"]["bg_period_s"]), labels, cfg)


def planted(monkeypatch, *spots: tuple[float, float], locked: list | None = None) -> None:
    """A stand-in for the blob finder: round dark blobs at these places whatever the picture.
    locked: what a tracker that is already on a ball gets to see instead."""
    cold = [Candidate(x, y, float(R), 1.0, True) for x, y in spots]
    near = cold if locked is None else [Candidate(x, y, float(R), 1.0, True) for x, y in locked]
    monkeypatch.setattr(tracker, "find_balls", lambda *a, **k: list(cold))
    monkeypatch.setattr(tracker, "round_blobs", lambda *a, **k: list(near))


def verdicts(marks: list) -> list[str]:
    return [m.verdict for m in marks]


def test_cold_step_finds_a_drawn_ball_on_every_frame(tmp_path):
    spots = [MID] * (WARM + 3)
    got = run(recorded(tmp_path, spots), lying(spots))
    assert got.frames == WARM + 3 and got.skipped == Counter({WARMING: WARM})
    for marks in got.cold, got.locked:
        assert verdicts(marks) == [CORRECT] * 3
        assert all(m.regions == (MIDDLE, ALL) and m.off_mm < 1.0 for m in marks)
    assert not got.unlocked


def test_a_stand_in_tracker_that_reports_nothing_is_scored_lost(tmp_path, monkeypatch):
    planted(monkeypatch)
    spots = [MID] * (WARM + 2)
    got = run(recorded(tmp_path, spots), lying(spots))
    assert verdicts(got.cold) == verdicts(got.locked) == [LOST] * 2
    assert all(m.off_mm is None for m in got.cold + got.locked)


def test_a_stand_in_tracker_that_reports_a_decoy_is_scored_wrong(tmp_path, monkeypatch):
    # a blob beside the ball, inside the locked tracker's search box
    planted(monkeypatch, (MID[0] + R + 6, MID[1]))
    spots = [MID] * (WARM + 2)
    got = run(recorded(tmp_path, spots), lying(spots))
    assert verdicts(got.cold) == verdicts(got.locked) == [WRONG] * 2
    assert all(m.off_mm is None for m in got.cold + got.locked)


def test_an_answer_inside_the_ball_is_measured_in_mm_and_put_in_the_balls_region(
    tmp_path, monkeypatch
):
    planted(monkeypatch, (RIGHT[0] + 3, RIGHT[1]))
    spots = [RIGHT] * (WARM + 1)
    got = run(recorded(tmp_path, spots), lying(spots))
    # 50 mm right of the middle of a board 60 mm to its edge, the answer 3 px = 1.5 mm off
    for (mark,) in got.cold, got.locked:
        assert mark == ((SIDES, EDGE, ALL), CORRECT, 1.5)


def test_the_region_of_a_moving_ball_is_where_it_was_half_way(tmp_path, monkeypatch):
    # from the middle of the board to 50 mm right of it: half way lies in the side third
    planted(monkeypatch, (250, 150))
    spots = [MID] * (WARM + 1)
    labels = {f: Label(MID, RIGHT, float(R), ARRIVAL, False, None) for f in lying(spots)}
    got = run(recorded(tmp_path, spots), labels)
    assert got.cold == [Mark((SIDES, ALL), CORRECT, None)]


def test_the_locked_step_starts_on_the_label_not_on_what_the_cold_find_took(tmp_path, monkeypatch):
    # from nothing the finder takes a blob across the board; next to the label lies the ball
    planted(monkeypatch, (60, 60), locked=[MID])
    spots = [MID] * (WARM + 2)
    got = run(recorded(tmp_path, spots), lying(spots))
    assert verdicts(got.cold) == [WRONG] * 2 and verdicts(got.locked) == [CORRECT] * 2


def test_the_kept_box_covers_the_whole_way_the_ball_went():
    moved = Label((100.0, 160.0), (200.0, 100.0), 20.0, ARRIVAL, False, None)
    assert replay._around(moved) == (60, 60, 240, 200)
    assert replay._around(moved._replace(a=moved.b, b=moved.a)) == (60, 60, 240, 200)


def test_total_adds_up_every_part():
    one = replay.Scored(
        5,
        Counter({WARMING: 2}),
        [Mark((MIDDLE, ALL), CORRECT, 0.5)] * 3,
        [Mark((MIDDLE, ALL), CORRECT, 0.25)],
        Counter({MOVING: 2}),
    )
    two = replay.Scored(
        4,
        Counter({WARMING: 1, UNSEEN: 1}),
        [Mark((SIDES, ALL), LOST, None)] * 2,
        [Mark((SIDES, ALL), WRONG, None)] * 2,
        Counter({NO_TONE: 1}),
    )
    assert replay.total([one, two]) == replay.Scored(
        9,
        Counter({WARMING: 3, UNSEEN: 1}),
        [*one.cold, *two.cold],
        [*one.locked, *two.locked],
        Counter({MOVING: 2, NO_TONE: 1}),
    )


def test_frames_a_second_apart_are_never_run_as_one_track(tmp_path):
    # the ball is across the board from one picture to the next: a tracker still locked on the
    # last picture would not follow it there
    spots = [MID] * WARM + [(80, 60), (320, 240), (80, 240)]
    got = run(recorded(tmp_path, spots), lying(spots))
    assert verdicts(got.cold) == verdicts(got.locked) == [CORRECT] * 3


def test_the_locked_step_is_scored_only_on_at_rest_frames(tmp_path):
    spots = [MID] * (WARM + 2)
    got = run(recorded(tmp_path, spots), lying(spots, at_rest=False))
    assert verdicts(got.cold) == [CORRECT] * 2 and all(m.off_mm is None for m in got.cold)
    assert got.locked == [] and got.unlocked == Counter({MOVING: 2})


def test_the_locked_step_is_not_run_when_the_ball_may_be_bright(tmp_path):
    spots = [MID] * (WARM + 2)
    folder = recorded(tmp_path, spots, FLAT + "ball_dark = false\n")
    got = run(folder, lying(spots))
    assert verdicts(got.cold) == [CORRECT] * 2
    assert got.locked == [] and got.unlocked == Counter({NO_TONE: 2})


def test_a_coasting_guess_is_not_an_answer(tmp_path):
    # the label says the ball lies where the picture shows nothing: the seeded tracker coasts on
    # the spot, which is a guess and not a blob it found
    spots = [MID] * WARM + [None]
    labels = lying([MID] * (WARM + 1))
    got = run(recorded(tmp_path, spots), labels)
    assert verdicts(got.cold) == verdicts(got.locked) == [LOST]
