import curses
import signal
import sys
import time

from src import safety
from src.contracts import ServoCommand
from src.control.pulse_map import center
from src.io.serial_link import ENVELOPE_US, BoardLink, LinkError, limits_line, open_board
from src.io.settings import load

NUDGE_US = 5


def headroom(rig: dict) -> str:
    lo, hi = ENVELOPE_US
    x_lo, x_hi, y_lo, y_hi = (round(v) for v in safety.limits(rig))
    return (
        f"headroom x: {x_lo - lo} below, {hi - x_hi} above"
        f" | y: {y_lo - lo} below, {hi - y_hi} above"
    )


def offset(pulses: tuple[int, int], rig: dict) -> str:
    cx, cy = center(rig)
    return f"off ({round(pulses[0] - cx):+d}, {round(pulses[1] - cy):+d}) us"


class Bench:
    def __init__(self, link: BoardLink, rig: dict):
        self.link = link
        self.rig = rig
        self.log: list[str] = []
        self.last_p: tuple[int, int] | None = None
        self.quiet = False
        self.killing: safety.SafetyState | None = None

    def _send(self, line: str, show: bool = True) -> str | None:
        try:
            reply = self.link.raw(line)
        except LinkError as e:
            self.log.append(f"{line}  ->  {e}")
            return None
        if show or not reply.startswith("OK"):
            self.log.append(f"{line}  ->  {reply}")
        return reply

    def submit(self, line: str) -> str:
        line = line.strip()
        if line == "quiet":
            self.quiet = True
            self.log.append("quiet: no more heartbeat")
            return ""
        words = line.split()
        if words[:1] == ["l"] and len(words) <= 2:
            return self._load(words)
        reply = self._send(line) or ""
        parts = reply.split()
        if parts[:2] == ["OK", "P"]:
            self.last_p = (int(parts[2]), int(parts[3]))
            self.quiet = False
        elif parts[:2] == ["OK", "D"] or "disarmed" in parts:
            self.last_p = None
        return reply

    def _load(self, words: list[str]) -> str:
        rig = self.rig
        if len(words) == 2:
            if not words[1].isdigit():
                self.log.append(f"{' '.join(words)} needs a whole number")
                return ""
            # bench-only limits, the settings file is untouched
            rng = int(words[1])
            rig = {**self.rig, "limit_x_us": rng, "limit_y_us": rng, "startup_flag": False}
        reply = self._send(limits_line(rig)) or ""
        if reply == "OK L":
            # the board took them, so they are the bench's limits now
            self.rig.update(rig)
            self.log.append(headroom(self.rig))
        return reply

    def nudge(self, dx: int, dy: int) -> None:
        if self.last_p is None or self.killing:
            return
        self.submit(f"P {self.last_p[0] + dx} {self.last_p[1] + dy}")

    def kill(self) -> None:
        if self.killing:
            return
        if self.last_p is None:
            self._send("D")
            return
        x, y = self.last_p
        self.killing = safety.SafetyState(armed=True, killing=True, x_us=x, y_us=y)
        self.log.append("KILL: ramping to level")

    def tick(self, dt: float) -> None:
        if self.killing:
            out = safety.step(self.killing, ServoCommand(*self.last_p), False, dt, self.rig)
            if out:
                self._send(f"P {out.x_us} {out.y_us}", show=False)
            if not self.killing.armed:
                self._send("D")
                self.killing = None
                self.last_p = None
        elif self.last_p and not self.quiet:
            self._send(f"P {self.last_p[0]} {self.last_p[1]}", show=False)


def kill_and_wait(bench: Bench, period: float) -> None:
    """Ramp to level and disarm; a second ctrl-c mid-ramp does nothing."""
    old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        bench.kill()
        while bench.killing:
            time.sleep(period)
            bench.tick(period)
    finally:
        signal.signal(signal.SIGINT, old)


def draw(scr, bench: Bench, line: str) -> None:
    h, w = scr.getmaxyx()
    scr.erase()
    state = "no pulses"
    if bench.last_p:
        state = f"pulses {bench.last_p} {offset(bench.last_p, bench.rig)}"
    if bench.killing:
        state = "KILLING"
    hb = "quiet" if bench.quiet else "heartbeat on"
    keys = f"space kill | l [range] | arrows nudge {NUDGE_US} us (lr x, ud y) | quiet | ctrl-c"
    scr.addnstr(0, 0, f"{state} | {hb} | {keys}", w - 1)
    for i, entry in enumerate(bench.log[-(h - 3) :]):
        scr.addnstr(i + 1, 0, entry, w - 1)
    scr.addnstr(h - 1, 0, "> " + line, w - 1)
    scr.refresh()


def run(scr, bench: Bench, period: float) -> None:
    arrows = {
        curses.KEY_LEFT: (-NUDGE_US, 0),
        curses.KEY_RIGHT: (NUDGE_US, 0),
        curses.KEY_UP: (0, NUDGE_US),
        curses.KEY_DOWN: (0, -NUDGE_US),
    }
    scr.nodelay(True)
    line = ""
    last = time.monotonic()
    try:
        while True:
            ch = scr.getch()
            if ch == ord(" ") and not line:
                bench.kill()
            elif ch in arrows:
                bench.nudge(*arrows[ch])
            elif ch in (curses.KEY_ENTER, 10, 13):
                if line:
                    bench.submit(line)
                line = ""
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                line = line[:-1]
            elif 32 <= ch < 127:
                line += chr(ch)
            now = time.monotonic()
            bench.tick(now - last)
            last = now
            draw(scr, bench, line)
            time.sleep(period)
    except BaseException as e:
        kill_and_wait(bench, period)
        if not isinstance(e, KeyboardInterrupt):
            raise


def main() -> None:
    cfg = load()
    rig = cfg["rig"]
    if len(sys.argv) > 1:
        rig["serial_port"] = sys.argv[1]
    bench = Bench(open_board(rig), rig)
    curses.wrapper(run, bench, 1 / cfg["control"]["sim_tick_hz"])


if __name__ == "__main__":
    main()
