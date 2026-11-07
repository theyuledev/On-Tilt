import serial

from src.contracts import PlateState, ServoCommand
from src.control.pulse_map import center, pulse_to_tilt
from src.safety import limits


class LinkError(RuntimeError):
    pass


def limits_line(rig: dict) -> str:
    cx, cy = center(rig)
    x_lo, x_hi, y_lo, y_hi = (round(v) for v in limits(rig))
    return f"L {x_lo} {x_hi} {y_lo} {y_hi} {round(cx)} {round(cy)}"


class BoardLink:
    """Speaks firmware/PROTOCOL.md over anything with write() and readline()."""

    def __init__(self, port):
        self.port = port

    def raw(self, line: str) -> str:
        self.port.write((line + "\n").encode())
        reply = self.port.readline().decode().strip()
        if not reply:
            raise LinkError(f"no reply to {line!r}")
        return reply

    def ask(self, line: str) -> list[str]:
        reply = self.raw(line)
        parts = reply.split()
        if parts[0] != "OK" or parts[1:2] != [line[0]]:
            raise LinkError(f"{line!r} -> {reply!r}")
        return parts[2:]

    def status(self) -> dict:
        armed, loaded, since_ms, fw = self.ask("S")
        return {
            "armed": armed == "1",
            "limits_loaded": loaded == "1",
            "since_ms": int(since_ms),
            "fw": fw,
        }


class SerialActuator:
    def __init__(self, link: BoardLink, rig: dict):
        self.link = link
        self.rig = rig
        self.armed = False
        self.state = PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0)

    def _arm(self) -> None:
        self.link.ask(limits_line(self.rig))
        self.link.ask("A")
        self.armed = True

    def send(self, cmd: ServoCommand | None, now: float) -> PlateState:
        if cmd is None:
            if self.armed:
                self.link.ask("D")
                self.armed = False
            return self.state

        if not self.armed:
            self._arm()
        x, y = (int(v) for v in self.link.ask(f"P {cmd.x_us} {cmd.y_us}"))
        tilt = pulse_to_tilt(ServoCommand(x_us=x, y_us=y), self.rig)
        self.state = PlateState(
            sent_x_deg=tilt.tilt_x_deg,
            sent_y_deg=tilt.tilt_y_deg,
            est_x_deg=tilt.tilt_x_deg,
            est_y_deg=tilt.tilt_y_deg,
        )
        return self.state


def open_board(rig: dict) -> BoardLink:
    port = rig["serial_port"]
    if port.startswith("TODO"):
        raise RuntimeError(f"serial_port not set yet: {port}")
    return BoardLink(serial.Serial(port, rig["serial_baud"], timeout=rig["serial_reply_timeout_s"]))
