import time

from src.contracts import Intent


class KeyboardInput:
    def __init__(self):
        self._intent = Intent(x=0.0, y=0.0, valid=False, t_s=0.0)

    def set(self, x: float, y: float) -> None:
        self._intent = Intent(x=x, y=y, valid=True, t_s=time.monotonic())

    def latest(self) -> Intent:
        return self._intent
