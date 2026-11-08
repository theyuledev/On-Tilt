import json
import math
import socket
from dataclasses import asdict

import numpy as np

from src.contracts import Intent

PALM = (0, 5, 9, 13, 17)  # wrist + the base of each finger
RECENTRE = b"c"


def palm_center(landmarks) -> tuple[float, float]:
    xs = [landmarks[i].x for i in PALM]
    ys = [landmarks[i].y for i in PALM]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def palm_angles(world) -> tuple[float, float]:
    """Roll and pitch in degrees of the palm normal, 0, 0 when the palm faces the camera."""
    o, a, b = (np.array([world[i].x, world[i].y, world[i].z]) for i in (0, 5, 17))
    n = np.cross(a - o, b - o)
    # the two hands wind the other way, so turn the normal to face the camera
    if n[2] > 0:
        n = -n
    return math.degrees(math.atan2(n[0], -n[2])), math.degrees(math.atan2(n[1], -n[2]))


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


class HandIntent:
    """Palm spot in the webcam image -> intent from -1 to 1."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.neutral: tuple[float, float] | None = None
        self.seen = False
        self._samples: list[tuple[float, float, float]] = []
        self._last_seen: float | None = None
        self._x = self._y = 0.0

    @property
    def calibrating(self) -> bool:
        return self.seen and self.neutral is None

    @property
    def progress(self) -> float:
        # how far through the hold still window, full once the neutral is set
        if self.neutral is not None:
            return 1.0
        if not self._samples:
            return 0.0
        return min(1.0, (self._samples[-1][0] - self._samples[0][0]) / self.cfg["neutral_s"])

    def update(self, palm: tuple[float, float] | None, t: float, recentre: bool = False) -> Intent:
        cfg = self.cfg
        self.seen = palm is not None
        if palm is None:
            gone = t - self._last_seen if self._last_seen is not None else 0.0
            if gone > cfg["neutral_s"]:
                self._samples = []
            # a short dropout keeps the neutral so the player does not hold still again
            if gone > cfg["neutral_keep_s"]:
                self.neutral = None
            return Intent(x=0.0, y=0.0, valid=False, t_s=t)

        self._last_seen = t
        if recentre:
            self._set_neutral(palm)
        if self.neutral is None:
            self._samples.append((t, *palm))
            if not self._lock_neutral(t):
                return Intent(x=0.0, y=0.0, valid=False, t_s=t)

        # the webcam faces the player, so their right is image left, and image y points down
        raw_x = _clamp(-(palm[0] - self.neutral[0]) / cfg["hand_span"])
        raw_y = _clamp(-(palm[1] - self.neutral[1]) / cfg["hand_span"])
        s = cfg["smoothing"]
        self._x = s * self._x + (1 - s) * raw_x
        self._y = s * self._y + (1 - s) * raw_y
        return Intent(x=self._x, y=self._y, valid=True, t_s=t)

    def _lock_neutral(self, t: float) -> bool:
        # a full window of a still hand sets neutral, a moving hand throws the window away
        if t - self._samples[0][0] < self.cfg["neutral_s"]:
            return False
        xs = [s[1] for s in self._samples]
        ys = [s[2] for s in self._samples]
        self._samples = []
        still = self.cfg["hand_span"] / 4
        if max(xs) - min(xs) >= still or max(ys) - min(ys) >= still:
            return False
        self._set_neutral((sum(xs) / len(xs), sum(ys) / len(ys)))
        return True

    def _set_neutral(self, at: tuple[float, float]) -> None:
        self.neutral = at
        self._samples = []
        self._x = self._y = 0.0


def recentre_asked(sock: socket.socket) -> bool:
    # the app answers on the sending socket when the player hits the re centre key
    asked = False
    while True:
        try:
            sock.recv(16)
        except BlockingIOError:
            return asked
        asked = True


def send_intent(
    sock: socket.socket,
    intent: Intent,
    port: int,
    *,
    seen: bool,
    calibrating: bool,
    progress: float = 0.0,
    lag_s: float | None = None,
) -> None:
    msg = {**asdict(intent), "seen": seen, "calibrating": calibrating, "progress": progress}
    if lag_s is not None:
        msg["lag_s"] = lag_s
    sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", port))


def _parse(data: bytes) -> tuple | None:
    try:
        msg = json.loads(data)
        if not all(math.isfinite(msg[k]) for k in ("x", "y", "t_s")):
            return None
        # an intent at all means a palm was there, older senders only send the intent
        flags = (
            msg.pop("seen", True),
            msg.pop("calibrating", False),
            msg.pop("progress", 0.0),
            msg.pop("lag_s", None),
        )
        # "false" is truthy, so only a real bool counts
        if not all(isinstance(v, bool) for v in (msg["valid"], *flags[:2])):
            return None
        return Intent(**msg), *flags
    except (ValueError, KeyError, TypeError, OverflowError):
        return None


class HandInput:
    """App side: keeps the newest intent the hand process sent."""

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self._intent = Intent(x=0.0, y=0.0, valid=False, t_s=0.0)
        self._peer = None
        self.seen = self.calibrating = False
        self.progress = 0.0
        self.lag_s: float | None = None
        self.dropped = 0

    def latest(self) -> Intent:
        while True:
            try:
                data, self._peer = self.sock.recvfrom(1024)
            except BlockingIOError:
                return self._intent
            parsed = _parse(data)
            if parsed is None:
                self.dropped += 1
                continue
            self._intent, self.seen, self.calibrating, self.progress, self.lag_s = parsed

    def recentre(self) -> None:
        if self._peer is not None:
            self.sock.sendto(RECENTRE, self._peer)

    def close(self) -> None:
        self.sock.close()
