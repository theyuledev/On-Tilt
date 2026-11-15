"""The line drawn on the board, read off the running app's own pictures, as the path to follow."""

import time

from src import switches
from src.contracts import Choices
from src.vision.line_path import MANY, LinePath, agree, find_line

LINE_READS = 5  # pictures the line is read off, the path most of them agree on is kept
ARMED, NO_PICTURE = "armed, kill first", "no camera picture"
# a refusal said as what to do about it; the others go to the page as the reader gives them
SAY = {MANY: "more than one line: put the ball on an end of the line you want"}


def read_line(loop, pipe) -> LinePath:
    """The drawn line off the next LINE_READS pictures, on the board map the tracker uses. Each
    picture goes with the ball seen at that moment: a ball on one end picks that line."""
    reads, last, rig = [], None, loop.cfg["rig"]
    while len(reads) < LINE_READS:
        frame, ball = loop.frame, loop.ball_px
        if loop.safety.armed:
            return LinePath([], ARMED)
        old = frame is None or time.monotonic() - frame.t_s > rig["stale_intent_s"]
        if pipe is None or old or loop.camera_lost:
            return LinePath([], NO_PICTURE)
        if frame is last:
            time.sleep(1 / rig["camera_fps"])
            continue
        last, edge = frame, pipe.cfg["line_edge_mm"]
        seen = ball if ball is not None and ball.valid else None
        reads.append(find_line(frame.image, pipe.cfg, pipe.half_mm, edge, seen))
    return agree(reads, pipe.cfg)


def use_line(loop, got: LinePath) -> str:
    """A path read becomes the one path mode follows; anything else changes nothing."""
    if loop.safety.armed:
        return ARMED
    if got.reason:
        return SAY.get(got.reason, got.reason)
    loop.cfg["control"]["path_mm"] = got.points
    switches.choose(loop, Choices(mode="path"))
    return "ok"
