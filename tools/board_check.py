import sys
import time
from contextlib import suppress

from src.control.pulse_map import center
from src.io.serial_link import ENVELOPE_US, BoardLink, LinkError, open_board
from src.io.settings import load

# the board drops out 250 ms after the last command: at level there is no ramp, then a 40 ms hold
QUIET_S = 0.35
SKIP_A = "limits already loaded this power cycle, replug to test"

Step = tuple[str, str, str | None]
Result = tuple[str, bool | None, str]


class Failed(Exception):
    pass


def steps(rig: dict, loaded: bool) -> tuple[list[Step], list[Step]]:
    cx, cy = (round(v) for v in center(rig))
    r = rig["startup_range_us"]
    h = r // 2
    ys = f"{cy - h} {cy + h}"
    before = [
        ("P while disarmed", f"P {cx} {cy}", "ERR P disarmed"),
        ("A before L", "A", None if loaded else "ERR A nolimits"),
        ("L outside envelope", f"L {ENVELOPE_US[0] - 1} {cx + h} {ys} {cx} {cy}", "ERR L envelope"),
        ("L inside startup range", f"L {cx - h} {cx + h} {ys} {cx} {cy}", "OK L"),
        ("arm", "A", "OK A"),
        ("P past the limit comes back clamped", f"P {cx + r} {cy}", f"OK P {cx + h} {cy}"),
        ("P at level", f"P {cx} {cy}", f"OK P {cx} {cy}"),
    ]
    after = [
        ("timeout leaves it disarmed", "S", "OK S 0"),
        ("re-arm after the timeout", "A", "OK A"),
        ("disarm", "D", "OK D"),
        ("disarmed after D", "S", "OK S 0"),
        ("marker led on", "M 1", "OK M"),
        ("marker led off", "M 0", "OK M"),
    ]
    return before, after


def send(link: BoardLink, rig: dict, line: str) -> str:
    words = line.split()
    if words[0] == "P":
        cx, cy = center(rig)
        if max(abs(int(words[1]) - cx), abs(int(words[2]) - cy)) > rig["startup_range_us"]:
            raise ValueError(f"refusing {line!r}, past startup_range_us from center")
    reply = link.raw(line)
    # S also carries ms since the last command and the fw version
    return " ".join(reply.split()[:3]) if line == "S" else reply


def run(link: BoardLink, rig: dict, todo: list[Step], out: list[Result]) -> None:
    for name, line, want in todo:
        if want is None:
            out.append((name, None, SKIP_A))
            continue
        try:
            got = send(link, rig, line)
        except (LinkError, OSError, ValueError) as e:
            out.append((name, False, str(e)))
            raise Failed from e
        out.append((name, got == want, f"{line!r} -> {got!r}, want {want!r}"))
        if got != want:
            raise Failed


def check(link: BoardLink, rig: dict, ask=input, sleep=time.sleep) -> list[Result]:
    try:
        parts = link.raw("S").split()
    except (LinkError, OSError) as e:
        return [("status", False, str(e))]
    if parts[:2] != ["OK", "S"] or len(parts) != 6:
        return [("status", False, f"'S' -> {' '.join(parts)!r}")]
    armed, loaded, _, fw = parts[2:]
    if armed != "0":
        return [(f"boots disarmed, fw {fw}", False, "armed, it should boot disarmed: replug it")]
    out: list[Result] = [(f"boots disarmed, fw {fw}", True, "")]
    if ask("servo power OFF? (y/N) ").strip().lower() != "y":
        return out + [("servo power off", False, "aborted, nothing sent")]
    before, after = steps(rig, loaded == "1")
    try:
        run(link, rig, before, out)
        sleep(QUIET_S)
        run(link, rig, after, out)
    except BaseException as e:
        with suppress(LinkError, OSError):
            link.raw("D")
        if not isinstance(e, Failed):
            raise
    return out


def show(results: list[Result]) -> bool:
    for name, ok, detail in results:
        if ok:
            print(f"ok  {name}")
        elif ok is None:
            print(f"skip {name}: {detail}")
        else:
            print(f"FAIL {name}: {detail}")
    return all(ok is not False for _, ok, _ in results)


def main() -> None:
    rig = load()["rig"]
    if len(sys.argv) > 1:
        rig["serial_port"] = sys.argv[1]
    if not show(check(open_board(rig), rig)):
        sys.exit(1)


if __name__ == "__main__":
    main()
