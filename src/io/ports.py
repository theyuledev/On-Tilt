from typing import NamedTuple, Protocol

import numpy as np

from src.contracts import Intent, PlateState, ServoCommand


class Frame(NamedTuple):
    image: np.ndarray
    t_s: float


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...

    def latest(self): ...

    def close(self) -> None: ...


class InputSource(Protocol):
    seen: bool
    calibrating: bool
    progress: float
    mode: str | None

    def latest(self) -> Intent: ...

    def recentre(self) -> None: ...

    def set_mode(self, mode: str) -> None: ...


class TiltActuator(Protocol):
    def send(self, cmd: ServoCommand | None, now: float) -> PlateState: ...
