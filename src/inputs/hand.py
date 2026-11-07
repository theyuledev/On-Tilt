import json
import socket
from dataclasses import asdict

from src.contracts import Intent

PALM = (0, 5, 9, 13, 17)  # wrist + the base of each finger


def palm_center(landmarks) -> tuple[float, float]:
    xs = [landmarks[i].x for i in PALM]
    ys = [landmarks[i].y for i in PALM]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


class HandIntent:
    """Palm spot in the webcam image -> intent from -1 to 1."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.neutral: tuple[float, float] | None = None
        self._samples: list[tuple[float, float, float]] = []
        self._last_seen: float | None = None
        self._x = self._y = 0.0

    def update(self, palm: tuple[float, float] | None, t: float) -> Intent:
        cfg = self.cfg
        if palm is None:
            if self._last_seen is not None and t - self._last_seen > cfg["neutral_s"]:
                self.neutral = None
                self._samples = []
            return Intent(x=0.0, y=0.0, valid=False, t_s=t)

        self._last_seen = t
        if self.neutral is None:
            self._samples.append((t, *palm))
            if t - self._samples[0][0] < cfg["neutral_s"]:
                return Intent(x=0.0, y=0.0, valid=False, t_s=t)
            n = len(self._samples)
            self.neutral = (
                sum(s[1] for s in self._samples) / n,
                sum(s[2] for s in self._samples) / n,
            )
            self._samples = []
            self._x = self._y = 0.0

        # the webcam faces the player, so their right is image left, and image y points down
        raw_x = _clamp(-(palm[0] - self.neutral[0]) / cfg["hand_span"])
        raw_y = _clamp(-(palm[1] - self.neutral[1]) / cfg["hand_span"])
        s = cfg["smoothing"]
        self._x = s * self._x + (1 - s) * raw_x
        self._y = s * self._y + (1 - s) * raw_y
        return Intent(x=self._x, y=self._y, valid=True, t_s=t)


def send_intent(sock: socket.socket, intent: Intent, port: int) -> None:
    sock.sendto(json.dumps(asdict(intent)).encode(), ("127.0.0.1", port))


class HandInput:
    """App side: keeps the newest intent the hand process sent."""

    def __init__(self, port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self._intent = Intent(x=0.0, y=0.0, valid=False, t_s=0.0)

    def latest(self) -> Intent:
        while True:
            try:
                data = self.sock.recv(1024)
            except BlockingIOError:
                return self._intent
            self._intent = Intent(**json.loads(data))

    def close(self) -> None:
        self.sock.close()
