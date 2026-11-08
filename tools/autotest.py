import math
import sys
import time
from contextlib import suppress
from itertools import pairwise, takewhile

from src.app import Loop, check_exposure
from src.contracts import BallState, Intent
from src.control.pulse_map import center
from src.io.camera import CameraSource, open_camera
from src.io.serial_link import BoardLink, LinkError, SerialActuator, open_board
from src.io.settings import load
from src.safety import LEVEL_HOLD_S, limits
from src.sim.fit import RELEASE_MM, hold_verdict
from src.vision.pipeline import Pipeline

TILT_US = 100
HOLD_S = 35.0
FIND_S = 5.0
# 250 ms board timeout + 100 us at 0.3 us/ms + 40 ms level hold, with margin
QUIET_S = 0.7

Result = tuple[str, str | None]


class Tap:
    """Keeps every line to and from the board."""

    def __init__(self, port):
        self.port, self.sent, self.replies = port, [], []

    def write(self, data: bytes) -> None:
        self.sent.append(data.decode().strip())
        self.port.write(data)

    def readline(self) -> bytes:
        reply = self.port.readline()
        self.replies.append(reply.decode().strip())
        return reply

    def reset_input_buffer(self) -> None:
        self.port.reset_input_buffer()


class Feed:
    intent = Intent(0.0, 0.0, valid=False, t_s=0.0)

    def latest(self) -> Intent:
        return self.intent


class Rig:
    def __init__(self, cfg, port, hold_loop, now=time.monotonic, sleep=time.sleep):
        self.cfg, self.r, self.hold_loop = cfg, cfg["rig"], hold_loop
        self.tap = Tap(port)
        self.link = BoardLink(self.tap)
        self.now, self.sleep = now, sleep
        self.dt = 1 / cfg["control"]["sim_tick_hz"]
        self.level = "P {} {}".format(*(round(v) for v in center(self.r)))
        # the longest kill ramp, from the edge of the limits
        x_lo, x_hi, y_lo, y_hi = limits(self.r)
        self.ramp_s = max(x_hi - x_lo, y_hi - y_lo) / 2 / self.r["kill_ramp_us_s"]

    def ticks(self, seconds: float) -> int:
        return math.ceil(seconds / self.dt)

    def tick(self, loop: Loop, x: float | None = None):
        self.sleep(self.dt)
        now = self.now()
        if x is not None:
            loop.keys.intent = Intent(x, 0.0, valid=True, t_s=now)
        return loop.tick(now, self.dt)

    def armed_is(self, want: bool) -> str | None:
        armed = self.link.status()["armed"]
        return None if armed == want else f"S says armed {int(armed)}"


def to_level(rig: Rig, loop: Loop) -> bool:
    loop.kill()
    for _ in range(rig.ticks(rig.ramp_s + LEVEL_HOLD_S) + 2):
        if not loop.safety.armed:
            break
        rig.tick(loop)
    rig.tick(loop)  # the D, once the level pulse went out
    return not loop.safety.armed


def tilt(rig: Rig, loop: Loop) -> None:
    loop.arm()
    cx = center(rig.r)[0]
    for _ in range(rig.ticks(1.0)):
        rec = rig.tick(loop, 1.0)
        if rec.armed and rec.servo_cmd.x_us - cx >= TILT_US:
            return
    raise LinkError(f"never got +{TILT_US} us, last {rec.servo_cmd}")


def ramp_problem(rig: Rig, lines: list[str]) -> str | None:
    ps = [s for s in lines if s.startswith("P ")]
    us = [[int(v) for v in s.split()[1:]] for s in ps]
    top = rig.r["kill_ramp_us_s"] * rig.dt + 1
    jump = max((abs(a - b) for p, q in pairwise(us) for a, b in zip(p, q, strict=True)), default=0)
    if jump > top:
        return f"ramp step {jump} us, max {top:g}"
    held = len(list(takewhile(lambda s: s == rig.level, reversed(ps))))
    if held < round(LEVEL_HOLD_S / rig.dt):
        return f"{held} level pulses before the stop"
    if lines.count("D") != 1 or lines[-1] != "D":
        return f"want one D after the level pulses, got {lines[-3:]}"
    return None


def kill(rig: Rig, loop: Loop) -> str | None:
    tilt(rig, loop)
    mark = len(rig.tap.sent) - 1
    if not to_level(rig, loop):
        return "still armed after the kill ramp"
    return ramp_problem(rig, rig.tap.sent[mark:]) or rig.armed_is(False)


def stale(rig: Rig, loop: Loop) -> str | None:
    tilt(rig, loop)
    start = len(rig.tap.sent)
    # ramp to level, then as long again holding there
    for _ in range(rig.ticks(rig.r["stale_intent_s"] + 2 * rig.ramp_s)):
        rig.tick(loop)
    off = [s for s in rig.tap.sent[-rig.ticks(rig.ramp_s) :] if s != rig.level]
    if "D" in rig.tap.sent[start:]:
        return "sent D on a stale intent"
    if off:
        return f"not holding level: {off[0]!r}"
    why = rig.armed_is(True)
    return why or (None if to_level(rig, loop) else "still armed after the kill ramp")


def timeout(rig: Rig, loop: Loop) -> str | None:
    tilt(rig, loop)
    rig.sleep(QUIET_S)
    s = rig.link.status()
    if s["armed"]:
        to_level(rig, loop)
        return f"still armed after {QUIET_S} s quiet"
    return f"since_ms {s['since_ms']}" if s["since_ms"] < round(QUIET_S * 1000) else None


def board_checks(rig: Rig) -> list[Result]:
    out = []
    for name, check in (("kill", kill), ("stale intent", stale), ("board timeout", timeout)):
        loop = Loop(rig.cfg, actuator=SerialActuator(rig.link, rig.r))
        loop.keys, loop.mode = Feed(), "hand"
        mark = len(rig.tap.replies)
        try:
            why = check(rig, loop)
        except BaseException as e:
            with suppress(LinkError, OSError, ValueError):
                to_level(rig, loop)
            if not isinstance(e, (LinkError, OSError, ValueError)):
                raise
            why = str(e)
        bad = [r for r in rig.tap.replies[mark:] if not r.startswith("OK")]
        out.append((name, why or (f"board said {bad[0] or 'nothing'!r}" if bad else None)))
    return out


def holding(rig: Rig, loop: Loop) -> str | None:
    # where assist stops short of the rim
    margin = rig.cfg["control"]["assist_margin_mm"]
    rim = (rig.r["plate_inner_mm"] - rig.r["ball_d_mm"]) / 2 - margin
    for _ in range(rig.ticks(HOLD_S)):
        rec = rig.tick(loop, 0.0)
        if not rec.armed:
            return "board lost" if loop.board_lost else "board disarmed"
        if not rec.ball.valid:
            return "ball lost"
        if math.hypot(rec.ball.x_mm, rec.ball.y_mm) > rim:
            return f"ball near the rim at ({rec.ball.x_mm:.0f}, {rec.ball.y_mm:.0f}) mm"
    return None


def hold(rig: Rig) -> list[Result]:
    loop = rig.hold_loop(rig)
    loop.keys, loop.mode = Feed(), "balance"
    for _ in range(rig.ticks(FIND_S)):
        if rig.tick(loop, 0.0).ball.valid:
            break
    trace = []
    loop.bus.subscribe("tick", trace.append)
    loop.arm()
    try:
        why = holding(rig, loop) if loop.safety.armed else f"arm refused: {loop.arm_blocked}"
    finally:
        to_level(rig, loop)
    if why:
        return [("balance hold", why)]
    rows = ((r.t_s, r.ball.x_mm, r.ball.y_mm) for r in trace[: rig.ticks(HOLD_S)])
    t, x, y = zip(*rows, strict=True)
    verdict = hold_verdict(t, x, y).items()
    return [(f"hold {k} {v:.2f}", None if ok else f"{v:.2f}") for k, (ok, v) in verdict]


def show(results: list[Result]) -> bool:
    for name, why in results:
        print(f"ok  {name}" if why is None else f"FAIL {name}: {why}")
    return all(why is None for _, why in results)


def sim_hold(rig: Rig) -> Loop:
    loop = Loop(rig.cfg)
    loop.ball = BallState(RELEASE_MM, 0.0, 0.0, 0.0, valid=True, t_s=rig.now())
    return loop


def real_hold(rig: Rig) -> Loop:
    frames = CameraSource(open_camera(rig.r), process=Pipeline(rig.cfg).step)
    actuator = SerialActuator(rig.link, rig.r)
    return Loop(rig.cfg, frames=frames, actuator=actuator, exposure=check_exposure(rig.r))


class Clock:
    t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def sim_rig(cfg: dict, board=None) -> Rig:
    # the timed fake board lives with the tests, real mode never needs it
    from tests.contract.timed_board import TimedBoard

    clock = Clock()
    return Rig(cfg, (board or TimedBoard)(clock.now), sim_hold, clock.now, clock.sleep)


def main() -> None:
    cfg = load()
    sim = "--sim" in sys.argv[1:]
    rig = sim_rig(cfg) if sim else Rig(cfg, open_board(cfg["rig"]).port, real_hold)
    if not show(board_checks(rig) + hold(rig)):
        sys.exit(1)


if __name__ == "__main__":
    main()
