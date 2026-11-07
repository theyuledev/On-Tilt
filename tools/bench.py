import curses
import sys
import time

from src import safety
from src.contracts import ServoCommand
from src.io.serial_link import BoardLink, LinkError, limits_line, open_board
from src.io.settings import load


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

    def submit(self, line: str) -> None:
        line = line.strip()
        if line == "quiet":
            self.quiet = True
            self.log.append("quiet: no more heartbeat")
            return
        if line == "l":
            line = limits_line(self.rig)
        reply = self._send(line) or ""
        parts = reply.split()
        if parts[:2] == ["OK", "P"]:
            self.last_p = (int(parts[2]), int(parts[3]))
            self.quiet = False
        elif parts[:2] == ["OK", "D"] or "disarmed" in parts:
            self.last_p = None

    def kill(self) -> None:
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


def draw(scr, bench: Bench, line: str) -> None:
    h, w = scr.getmaxyx()
    scr.erase()
    state = f"pulses {bench.last_p}" if bench.last_p else "no pulses"
    if bench.killing:
        state = "KILLING"
    hb = "quiet" if bench.quiet else "heartbeat on"
    keys = "space kill | l limits | quiet | ctrl-c"
    scr.addnstr(0, 0, f"{state} | {hb} | {keys}", w - 1)
    for i, entry in enumerate(bench.log[-(h - 3) :]):
        scr.addnstr(i + 1, 0, entry, w - 1)
    scr.addnstr(h - 1, 0, "> " + line, w - 1)
    scr.refresh()


def run(scr, bench: Bench, period: float) -> None:
    scr.nodelay(True)
    line = ""
    last = time.monotonic()
    try:
        while True:
            ch = scr.getch()
            if ch == ord(" ") and not line:
                bench.kill()
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
    except KeyboardInterrupt:
        bench.kill()
        while bench.killing:
            time.sleep(period)
            bench.tick(period)


def main() -> None:
    cfg = load()
    rig = cfg["rig"]
    if len(sys.argv) > 1:
        rig["serial_port"] = sys.argv[1]
    bench = Bench(open_board(rig), rig)
    curses.wrapper(run, bench, 1 / cfg["control"]["sim_tick_hz"])


if __name__ == "__main__":
    main()
