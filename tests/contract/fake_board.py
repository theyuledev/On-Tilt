ENVELOPE = (1050, 1950)


class FakeBoard:
    """Acts like firmware/tilt over a serial port, minus timing."""

    def __init__(self):
        self.armed = False
        self.limits = None
        self.pulses = None
        self.led = False
        self.silent = False
        self.sent: list[str] = []
        self._replies: list[str] = []

    def write(self, data: bytes) -> None:
        line = data.decode().strip()
        self.sent.append(line)
        if not self.silent:
            self._replies.append(self._handle(line.split()))

    def readline(self) -> bytes:
        return (self._replies.pop(0) + "\n").encode() if self._replies else b""

    def _handle(self, parts: list[str]) -> str:
        if not parts or len(parts[0]) != 1:
            return "ERR ?"
        cmd, args = parts[0], [int(a) for a in parts[1:]]

        if cmd == "S" and not args:
            return f"OK S {int(self.armed)} {int(self.limits is not None)} 0 fake"
        if cmd == "L" and len(args) == 6:
            if self.armed:
                return "ERR L armed"
            x_lo, x_hi, y_lo, y_hi, x_lvl, y_lvl = args
            ok = all(ENVELOPE[0] <= v <= ENVELOPE[1] for v in args)
            ok = ok and x_lo < x_hi and y_lo < y_hi
            ok = ok and x_lo <= x_lvl <= x_hi and y_lo <= y_lvl <= y_hi
            if not ok:
                return "ERR L envelope"
            self.limits = args
            return "OK L"
        if cmd == "A" and not args:
            if self.limits is None:
                return "ERR A nolimits"
            if not self.armed:
                self.armed = True
                self.pulses = (self.limits[4], self.limits[5])
            return "OK A"
        if cmd == "D" and not args:
            self.armed = False
            self.pulses = None
            return "OK D"
        if cmd == "P" and len(args) == 2:
            if not self.armed:
                return "ERR P disarmed"
            x_lo, x_hi, y_lo, y_hi = self.limits[:4]
            self.pulses = (min(max(args[0], x_lo), x_hi), min(max(args[1], y_lo), y_hi))
            return f"OK P {self.pulses[0]} {self.pulses[1]}"
        if cmd == "M" and len(args) == 1:
            self.led = bool(args[0])
            return "OK M"
        return "ERR ?"
