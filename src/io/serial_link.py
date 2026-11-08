import termios

import serial

from src.contracts import PlateState, ServoCommand
from src.control.pulse_map import center, pulse_to_tilt
from src.safety import limits

ENVELOPE_US = (1050, 1950)  # firmware refuses anything outside
STRAGGLERS = 2  # late replies to earlier commands that may turn up before this one's


class LinkError(RuntimeError):
    pass


class BoardRefused(LinkError):
    """The board answered ERR, so its state is known."""


def limits_line(rig: dict) -> str:
    cx, cy = center(rig)
    x_lo, x_hi, y_lo, y_hi = (round(v) for v in limits(rig))
    return f"L {x_lo} {x_hi} {y_lo} {y_hi} {round(cx)} {round(cy)}"


class BoardLink:
    """Speaks firmware/PROTOCOL.md over anything with write(), readline(), reset_input_buffer()."""

    def __init__(self, port):
        self.port = port

    def flush(self) -> None:
        # a reply that landed after the timeout is still queued and would be read as the next one's
        try:
            self.port.reset_input_buffer()
        except termios.error as e:
            # gone port, and termios.error is not an OSError on its own
            raise OSError(str(e)) from e

    def _read(self, line: str) -> str:
        reply = self.port.readline().decode().strip()
        if not reply:
            raise LinkError(f"no reply to {line!r}")
        return reply

    def raw(self, line: str) -> str:
        self.port.write((line + "\n").encode())
        return self._read(line)

    def ask(self, line: str) -> list[str]:
        self.port.write((line + "\n").encode())
        for _ in range(STRAGGLERS + 1):
            reply = self._read(line)
            parts = reply.split()
            if parts[0] == "ERR":
                raise BoardRefused(f"{line!r} -> {reply!r}")
            if parts[0] == "OK" and parts[1:2] == [line[0]]:
                return parts[2:]
        raise LinkError(f"{line!r} -> {reply!r}")

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
        self.acked = True  # the last P got its OK back
        self.state = PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0)

    def _arm(self) -> None:
        self.link.flush()
        self.link.ask(limits_line(self.rig))
        self.link.ask("A")
        self.armed = True

    def still_armed(self) -> bool:
        self.link.flush()
        return self.link.status()["armed"]

    def send(self, cmd: ServoCommand | None, now: float) -> PlateState:
        if cmd is None:
            if self.armed:
                # after an unacknowledged P the board may be mid ramp, a D there detaches
                # off level; its own timeout levels it and stops pulses instead
                if self.acked:
                    self.link.ask("D")
                self.armed = False
            return self.state

        if not self.armed:
            self._arm()
        try:
            x, y = (int(v) for v in self.link.ask(f"P {cmd.x_us} {cmd.y_us}"))
        except LinkError as e:
            # only a timeout leaves the board in doubt; an ERR is its own word, during its
            # self ramp it still says armed=1 so asking S there would keep us going
            if isinstance(e, BoardRefused) or not self.still_armed():
                raise
            self.acked = False
            return self.state
        self.acked = True
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
