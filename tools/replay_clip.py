"""Hit rate of the ball tracker over a recorded clip.

    uv run python -m tools.replay_clip data/clips/<clip>.avi [start:end ...]

Windows are frame numbers, end excluded. No window means the whole clip.
"""

import sys
from statistics import fmean
from time import perf_counter
from typing import NamedTuple

import numpy as np

from src.io.camera import ClipSource
from src.io.settings import load
from src.vision.tracker import BallTracker

Window = tuple[int, int]


class Replay(NamedTuple):
    """Per window: radius of every frame the tracker saw the ball in, and the frames it missed.
    Plus the cost of every update in ms, split by whether the tracker was locked or searching."""

    windows: dict[Window, tuple[list[float], list[int]]]
    locked_ms: list[float]
    search_ms: list[float]


def parse_window(arg: str) -> Window:
    a, b = arg.split(":")
    return int(a), int(b)


def run(path: str, windows: list[Window], cfg: dict, start: int = 0) -> Replay:
    src = ClipSource(path, start)
    tracker = BallTracker(cfg)
    radii: dict[Window, list[float]] = {w: [] for w in windows}
    missed: dict[Window, list[int]] = {w: [] for w in windows}
    ms: dict[bool, list[float]] = {True: [], False: []}
    last = max(end for _, end in windows)
    i = start
    while i < last and (frame := src.read()) is not None:
        locked = tracker.r is not None
        t0 = perf_counter()
        out = tracker.update(frame.image, frame.t_s)
        ms[locked].append((perf_counter() - t0) * 1e3)
        seen = out.valid and tracker.misses == 0
        for w in windows:
            if w[0] <= i < w[1]:
                (radii[w].append(out.r_px) if seen else missed[w].append(i))
        i += 1
    src.close()
    return Replay({w: (radii[w], missed[w]) for w in windows}, ms[True], ms[False])


def pct(ms: list[float], q: float) -> float:
    return float(np.percentile(ms, q)) if ms else 0.0


def spans(frames: list[int]) -> str:
    out: list[list[int]] = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return " ".join(str(a) if a == b else f"{a}-{b}" for a, b in out) or "-"


def main(argv: list[str]) -> None:
    path, *args = argv
    windows = [parse_window(a) for a in args] or [(0, sys.maxsize)]
    res = run(path, windows, load()["vision"])
    for (a, b), (radii, missed) in res.windows.items():
        n = len(radii) + len(missed)
        rate = len(radii) / n if n else 0.0
        r = fmean(radii) if radii else 0.0
        end = min(b, a + n)
        print(f"{a}:{end} frames {n} hit {rate:.3f} mean r {r:.1f} missed {spans(missed)}")
    for name, ms in ("locked", res.locked_ms), ("search", res.search_ms):
        print(f"{name} {len(ms)} frames p50 {pct(ms, 50):.2f} ms p99 {pct(ms, 99):.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1:])
