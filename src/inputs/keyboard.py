import time

from src.contracts import Intent


class KeyboardInput:
    """Held keys ease in over key_ramp_s so a press does not slam the plate."""

    def __init__(self, cfg: dict):
        self.ramp_s = cfg["key_ramp_s"]
        self._intent = Intent(x=0.0, y=0.0, valid=False, t_s=0.0)
        self._target = (0.0, 0.0)
        self._since = (0.0, 0.0)

    def set(self, x: float, y: float) -> None:
        now = time.monotonic()
        tx, ty = self._target
        sx, sy = self._since
        # an axis restarts its ramp whenever its keys change, so a release is instant
        self._since = (sx if x == tx else now, sy if y == ty else now)
        self._target = (x, y)
        self._intent = Intent(
            x=self._ramped(x, self._since[0], now),
            y=self._ramped(y, self._since[1], now),
            valid=True,
            t_s=now,
        )

    def _ramped(self, v: float, since: float, now: float) -> float:
        if self.ramp_s <= 0:
            return v
        return v * min(1.0, (now - since) / self.ramp_s)

    def latest(self) -> Intent:
        return self._intent
