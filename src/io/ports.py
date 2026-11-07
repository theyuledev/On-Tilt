from typing import NamedTuple, Protocol

import numpy as np


class Frame(NamedTuple):
    image: np.ndarray
    t_s: float


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...

    def close(self) -> None: ...
