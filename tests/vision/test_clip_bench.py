"""Ball tracker on the recorded clips. By default only the floors and the update cost, each window
run on its own after a short warm-up (about 15 s for the three clips). VISION_BENCH=full also runs
the nine ratchet windows from frame 0 against bench_baseline.json (about 3 min)."""

import json
import os
from pathlib import Path

import pytest

from src.io.camera import ClipSource
from src.io.settings import ROOT, load
from src.vision.tracker import BG_MIN_SAMPLES, BallTracker
from tools.replay_clip import Window, parse_window, pct, run

pytestmark = pytest.mark.replay

CLIPS = ROOT / "data/clips"
CFG = load()["vision"]
# hit rate per window from the last pr that moved the detector; whoever raises one rewrites it
BASELINE: dict[str, dict[str, float]] = json.loads(
    Path(__file__).with_name("bench_baseline.json").read_text()
)
# windows where the ball is in view the whole time: lacrosse rolls along the edge with the hand
# nudging it 13386-13408, squash rolls free
FLOORS: dict[tuple[str, Window], float] = {
    ("lacrosse_ball_hand_rolled_1948", (13310, 13430)): 0.85,
    ("squash_ball_hand_rolled_1944", (5130, 5370)): 0.98,
}
# frames before a floor window: 10 s at 120 fps, enough for the background's 5 samples
WARMUP = 1200

if not any((CLIPS / f"{stem}.avi").exists() for stem in BASELINE):
    pytest.skip(f"no clips in {CLIPS}", allow_module_level=True)


def hit_rate(window: tuple[list[float], list[int]]) -> float:
    radii, missed = window
    return len(radii) / (len(radii) + len(missed))


def pick(results: dict, key, stem: str):
    if key not in results:
        pytest.skip(f"{stem}.avi not on this machine")
    return results[key]


@pytest.fixture(scope="module")
def floors():
    out = {}
    for stem, window in FLOORS:
        path = CLIPS / f"{stem}.avi"
        if path.exists():
            out[stem, window] = run(str(path), [window], CFG, window[0] - WARMUP)
    return out


@pytest.fixture(scope="module")
def runs():
    if os.environ.get("VISION_BENCH") != "full":
        pytest.skip("set VISION_BENCH=full")
    out = {}
    for stem, windows in BASELINE.items():
        path = CLIPS / f"{stem}.avi"
        if path.exists():
            out[stem] = run(str(path), [parse_window(k) for k in windows], CFG)
    return out


@pytest.mark.parametrize(("stem", "window", "floor"), [(s, w, f) for (s, w), f in FLOORS.items()])
def test_visible_ball_is_held(floors, stem, window, floor):
    rate = hit_rate(pick(floors, (stem, window), stem).windows[window])
    print(f"{stem} {window} hit {rate:.3f} floor {floor}")
    assert rate >= floor, f"{stem} {window}: hit {rate:.3f} < {floor}"


def test_update_cost(floors):
    locked = [ms for r in floors.values() for ms in r.locked_ms]
    search = [ms for r in floors.values() for ms in r.search_ms]
    print(f"locked p50 {pct(locked, 50):.2f} p99 {pct(locked, 99):.2f} ms", end=" ")
    print(f"search p50 {pct(search, 50):.2f} p99 {pct(search, 99):.2f} ms")
    assert pct(locked, 50) <= 2.0, "locked update usually fits a 120 fps budget with room to spare"
    assert pct(locked, 99) <= 5.0, "a background sample is the only thing allowed to cost more"
    # 1.5x the 12.8 ms p99 measured on a busy mac, the aim is 12
    assert pct(search, 99) <= 19.0, "a full frame search while lost returns within two frames"


@pytest.mark.parametrize(("stem", "key"), [(s, k) for s in BASELINE for k in BASELINE[s]])
def test_hit_rate_holds_the_baseline(runs, stem, key):
    rate = hit_rate(pick(runs, stem, stem).windows[parse_window(key)])
    base = BASELINE[stem][key]
    print(f"{stem} {key} hit {rate:.3f} baseline {base:.3f}")
    assert rate >= base - 0.005, f"{stem} {key}: hit {rate:.3f} dropped under baseline {base:.3f}"


def test_empty_plate_gives_no_detections():
    clips = sorted(CLIPS.glob("empty_plate*.avi"))
    if not clips:
        pytest.skip("no empty plate clip yet")
    warm_s = BG_MIN_SAMPLES * CFG["bg_period_s"]
    for path in clips:
        tracker = BallTracker(CFG)
        src = ClipSource(str(path))
        t0 = t = None
        hits = runs = 0
        prev = False
        while (frame := src.read()) is not None:
            out = tracker.update(frame.image, frame.t_s)
            t0 = frame.t_s if t0 is None else t0
            t = frame.t_s
            if t - t0 < warm_s:
                continue
            # a detection, not a coasted frame
            hit = out.valid and tracker.misses == 0
            hits += hit
            runs += hit and not prev
            prev = hit
        src.close()
        minutes = (t - t0 - warm_s) / 60
        print(f"{path.name}: {hits} detections in {runs} runs over {minutes:.1f} min")
        assert hits == 0, f"{path.name}: {hits / minutes:.1f} false detections per minute"
