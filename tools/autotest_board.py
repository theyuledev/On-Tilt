import math
import time
from contextlib import suppress
from itertools import pairwise, takewhile

from src.contracts import Intent
from src.control.pulse_map import center
from src.io.serial_link import BoardLink, LinkError, SerialActuator
from src.loop import Loop
from src.safety import LEVEL_HOLD_S, limits

TILT_US = 100
# firmware/tilt: 250 ms timeout, then a 0.3 us/ms ramp to level and a 40 ms hold; plus a margin
FW_TIMEOUT_S, FW_RAMP_US_S, FW_HOLD_S, QUIET_MARGIN_S = 0.25, 300.0, 0.04, 0.1

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


def tilt(rig: Rig, loop: Loop) -> float:
    loop.arm()
    cx, cy = center(rig.r)
    for _ in range(rig.ticks(1.0)):
        rec = rig.tick(loop, 1.0)
        x, y = rec.servo_cmd.x_us, rec.servo_cmd.y_us
        if rec.armed and x - cx >= TILT_US:
            return max(abs(x - cx), abs(y - cy))
    raise LinkError(f"never got +{TILT_US} us, last {rec.servo_cmd}")


def quiet_s(off_us: float) -> float:
    return FW_TIMEOUT_S + off_us / FW_RAMP_US_S + FW_HOLD_S + QUIET_MARGIN_S


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
    quiet = quiet_s(tilt(rig, loop))
    rig.sleep(quiet)
    s = rig.link.status()
    if s["armed"]:
        to_level(rig, loop)
        return f"still armed after {quiet:.2f} s quiet"
    return f"since_ms {s['since_ms']}" if s["since_ms"] < round(quiet * 1000) else None


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
