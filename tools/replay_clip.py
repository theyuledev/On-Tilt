"""Hit rate of the ball tracker over a recorded clip.

    uv run python -m tools.replay_clip data/clips/<clip>.avi [start:end ...]

Windows are frame numbers, end excluded. No window means the whole clip.
"""

import sys
from statistics import fmean

from src.io.camera import ClipSource
from src.io.settings import load
from src.vision.tracker import BallTracker

Window = tuple[int, int]


def parse_window(arg: str) -> Window:
    a, b = arg.split(":")
    return int(a), int(b)


def run(path: str, windows: list[Window], cfg: dict) -> dict[Window, tuple[list[float], list[int]]]:
    """Per window: radius of every frame the tracker saw the ball in, and the frames it missed."""
    src = ClipSource(path)
    tracker = BallTracker(cfg)
    radii: dict[Window, list[float]] = {w: [] for w in windows}
    missed: dict[Window, list[int]] = {w: [] for w in windows}
    last = max(end for _, end in windows)
    i = 0
    while i < last and (frame := src.read()) is not None:
        out = tracker.update(frame.image, frame.t_s)
        seen = out.valid and tracker.misses == 0
        for w in windows:
            if w[0] <= i < w[1]:
                (radii[w].append(out.r_px) if seen else missed[w].append(i))
        i += 1
    src.close()
    return {w: (radii[w], missed[w]) for w in windows}


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
    for (a, b), (radii, missed) in run(path, windows, load()["vision"]).items():
        n = len(radii) + len(missed)
        rate = len(radii) / n if n else 0.0
        r = fmean(radii) if radii else 0.0
        end = min(b, a + n)
        print(f"{a}:{end} frames {n} hit {rate:.3f} mean r {r:.1f} missed {spans(missed)}")


if __name__ == "__main__":
    main(sys.argv[1:])
