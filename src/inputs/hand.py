import json
import math
import socket
from collections import deque
from dataclasses import asdict
from statistics import median

from src.contracts import Intent
from src.inputs.one_euro import one_euro
from src.inputs.palm import PALM, palm_angles, palm_center  # noqa: F401

RECENTRE = b"c"
SPAN = {"tilt": "hand_tilt_span_deg", "palm": "hand_span"}


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _signs(cfg: dict) -> tuple[float, float]:
    # webcam above the hand looking down, image top away from the player, not mirrored:
    # tilt: right side down reads roll +, fingertips down reads pitch -
    # palm: image x is the player's right, image y points back at the player
    return (-1.0 if cfg["hand_flip_x"] else 1.0), (1.0 if cfg["hand_flip_y"] else -1.0)


class HandIntent:
    """Palm tilt in degrees or palm spot in the image -> intent from -1 to 1."""

    def __init__(self, cfg: dict, *, stale_s: float | None = None):
        self.cfg = cfg
        self.stale_s = cfg["neutral_s"] if stale_s is None else stale_s
        self.seen = False
        self.set_mode(cfg["hand_mode"])

    def set_mode(self, mode: str) -> None:
        # tilt reads degrees and palm reads image fractions, so neutral and filter start over
        self.mode, self.span = mode, self.cfg[SPAN[mode]]
        self.neutral: tuple[float, float] | None = None
        self._samples: list[tuple[float, float, float]] = []
        self._last_seen: float | None = None
        self._last_capture: float | None = None
        self._euro = (None, None)
        self._readings = deque(maxlen=3)
        self._appear = 0.0

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

    def update(
        self, reading: tuple[float, float] | None, t: float, recentre: bool = False
    ) -> Intent:
        cfg = self.cfg
        if not math.isfinite(t):
            return Intent(0.0, 0.0, False, self._last_seen or 0.0)
        if self._last_capture is not None and t <= self._last_capture:
            return Intent(0.0, 0.0, False, t)
        self._last_capture = t
        if reading is not None and not all(math.isfinite(v) for v in reading):
            reading = None
        if reading is not None and self._last_seen is not None:
            if t - self._last_seen > self.stale_s:
                self._lost(t)
                self._samples = []
        was_seen, self.seen = self.seen, reading is not None
        if reading is None:
            return self._lost(t)

        self._last_seen = t
        if not was_seen:
            self._appear = t
        if recentre:
            self._euro = (None, None)
            self._readings.clear()
        at = self._smooth(reading, t)
        if recentre:
            self._set_neutral(at, t)
        if self.neutral is None:
            self._samples.append((t, *at))
            if not self._lock_neutral(t):
                return Intent(x=0.0, y=0.0, valid=False, t_s=t)

        sx, sy = _signs(cfg)
        # ease in after the hand shows up or the neutral moves, so the plate never jumps
        ramp_s = cfg["key_ramp_s"]
        ramp = min(1.0, (t - self._appear) / ramp_s) if ramp_s > 0 else 1.0
        x = _clamp(sx * (at[0] - self.neutral[0]) / self.span) * ramp
        y = _clamp(sy * (at[1] - self.neutral[1]) / self.span) * ramp
        return Intent(x=x, y=y, valid=True, t_s=t)

    def _lost(self, t: float) -> Intent:
        self.seen = False
        self._euro = (None, None)
        self._readings.clear()
        gone = t - self._last_seen if self._last_seen is not None else 0.0
        if gone > self.cfg["neutral_s"]:
            self._samples = []
        # a short dropout keeps the neutral so the player does not hold still again
        if gone > self.cfg["neutral_keep_s"]:
            self.neutral = None
        return Intent(x=0.0, y=0.0, valid=False, t_s=t)

    def _smooth(self, reading: tuple[float, float], t: float) -> tuple[float, float]:
        self._readings.append(reading)
        if len(self._readings) == self._readings.maxlen:
            reading = tuple(median(vs) for vs in zip(*self._readings, strict=True))
        # filtered in spans so hand_beta means the same in both modes
        out = [
            one_euro(s, v / self.span, t, self.cfg)
            for s, v in zip(self._euro, reading, strict=True)
        ]
        self._euro = tuple(s for _, s in out)
        return out[0][0] * self.span, out[1][0] * self.span

    def _lock_neutral(self, t: float) -> bool:
        # a full window of a still hand sets neutral, a moving hand throws the window away
        if t - self._samples[0][0] < self.cfg["neutral_s"]:
            return False
        xs = [s[1] for s in self._samples]
        ys = [s[2] for s in self._samples]
        self._samples = []
        still = self.span / 4
        if max(xs) - min(xs) >= still or max(ys) - min(ys) >= still:
            return False
        self._set_neutral((sum(xs) / len(xs), sum(ys) / len(ys)), t)
        return True

    def _set_neutral(self, at: tuple[float, float], t: float) -> None:
        self.neutral = at
        self._samples = []
        self._appear = t


def asked(sock: socket.socket) -> list[bytes]:
    # the app answers on the sending socket: the re centre key, or a hand mode by name
    msgs = []
    while True:
        try:
            msgs.append(sock.recv(16))
        except BlockingIOError:
            return msgs


def send_intent(
    sock: socket.socket,
    intent: Intent,
    port: int,
    *,
    seen: bool,
    calibrating: bool,
    progress: float = 0.0,
    lag_s: float | None = None,
    mode: str | None = None,
) -> None:
    msg = {**asdict(intent), "seen": seen, "calibrating": calibrating, "progress": progress}
    if lag_s is not None:
        msg["lag_s"] = lag_s
    if mode is not None:
        msg["mode"] = mode
    sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", port))


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
        self.mode: str | None = None  # the hand mode wanted, the sender's own until one is set

    def latest(self) -> Intent:
        while True:
            try:
                data, self._peer = self.sock.recvfrom(1024)
            except BlockingIOError:
                return self._intent
            msg = json.loads(data)
            if not (math.isfinite(msg["x"]) and math.isfinite(msg["y"])):
                continue
            # an intent at all means a palm was there, older senders only send the intent
            self.seen = msg.pop("seen", True)
            self.calibrating = msg.pop("calibrating", False)
            self.progress = msg.pop("progress", 0.0)
            self.lag_s = msg.pop("lag_s", None)
            self._heard(msg.pop("mode", None))
            self._intent = Intent(**msg)

    def _heard(self, mode: str | None) -> None:
        if self.mode is None:
            self.mode = mode
        elif mode is not None and mode != self.mode:
            # a restarted or deaf sender is told again until it switches
            self._tell(self.mode.encode())

    def _tell(self, data: bytes) -> None:
        if self._peer is not None:
            self.sock.sendto(data, self._peer)

    def recentre(self) -> None:
        self._tell(RECENTRE)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._tell(mode.encode())

    def close(self) -> None:
        self.sock.close()
