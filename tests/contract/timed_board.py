from tests.contract.fake_board import FakeBoard
from tools.autotest_board import FW_HOLD_S, FW_RAMP_US_S, FW_TIMEOUT_S


class TimedBoard(FakeBoard):
    """FakeBoard on a clock: quiet past the timeout it ramps itself to level, holds, disarms."""

    def __init__(self, now):
        super().__init__()
        self.now = now
        self.last = now()
        self.since_ms = 0

    def write(self, data: bytes) -> None:
        t = self.now()
        if self.armed:
            cx, cy = self.limits[4:]
            off = max(abs(self.pulses[0] - cx), abs(self.pulses[1] - cy))
            if t - self.last > FW_TIMEOUT_S + off / FW_RAMP_US_S + FW_HOLD_S:
                self.armed, self.pulses = False, None
        self.since_ms, self.last = round((t - self.last) * 1000), t
        super().write(data)

    def _handle(self, parts: list[str]) -> str:
        reply = super()._handle(parts)
        if parts == ["S"]:
            words = reply.split()
            words[4] = str(self.since_ms)
            reply = " ".join(words)
        return reply
