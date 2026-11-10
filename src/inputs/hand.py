import json
import math
import socket
from dataclasses import asdict

import numpy as np

from src.contracts import Intent
from src.inputs.one_euro import one_euro

PALM = (0, 5, 9, 13, 17)  # wrist + the base of each finger
RECENTRE = b"c"
SPAN = {"tilt": "hand_tilt_span_deg", "palm": "hand_span"}


def palm_center(landmarks) -> tuple[float, float]:
    xs = [landmarks[i].x for i in PALM]
    ys = [landmarks[i].y for i in PALM]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def palm_angles(world) -> tuple[float, float] | None:
    """Roll and pitch in degrees of the palm normal, 0, 0 when the palm faces the camera."""
    o, a, b = (np.array([world[i].x, world[i].y, world[i].z]) for i in (0, 5, 17))
    n = np.cross(a - o, b - o)
    if np.linalg.norm(n) == 0:
        return None
    # the two hands wind the other way, so turn the normal to face the camera
    if n[2] > 0:
        n = -n
    return math.degrees(math.atan2(n[0], -n[2])), math.degrees(math.atan2(n[1], -n[2]))


def _clamp(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _signs(cfg: dict) -> tuple[float, float]:
    # webcam above the hand looking down, image top away from the player, not mirrored:
    # tilt: right side down reads roll +, fingertips down reads pitch -
    # palm: image x is the player's right, image y points back at the player
    return (-1.0 if cfg["hand_flip_x"] else 1.0), (1.0 if cfg["hand_flip_y"] else -1.0)


class HandIntent:
    """Palm tilt in degrees or palm spot in the image -> intent from -1 to 1."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seen = False
        self.set_mode(cfg["hand_mode"])

    def set_mode(self, mode: str) -> None:
        # tilt reads degrees and palm reads image fractions, so neutral and filter start over
        self.mode, self.span = mode, self.cfg[SPAN[mode]]
        self.neutral: tuple[float, float] | None = None
        self._samples: list[tuple[float, float, float]] = []
        self._last_seen: float | None = None
        self._euro = (None, None)
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
        was_seen, self.seen = self.seen, reading is not None
        if reading is None:
            self._euro = (None, None)
            gone = t - self._last_seen if self._last_seen is not None else 0.0
            if gone > cfg["neutral_s"]:
                self._samples = []
            # a short dropout keeps the neutral so the player does not hold still again
            if gone > cfg["neutral_keep_s"]:
                self.neutral = None
            return Intent(x=0.0, y=0.0, valid=False, t_s=t)

        self._last_seen = t
        if not was_seen:
            self._appear = t
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

    def _smooth(self, reading: tuple[float, float], t: float) -> tuple[float, float]:
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
