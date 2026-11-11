import math
from collections import deque

from src.contracts import Intent


def _bounded(intent: Intent) -> Intent:
    scale = max(1.0, math.hypot(intent.x, intent.y))
    return Intent(intent.x / scale, intent.y / scale, intent.valid, intent.t_s)


class HandPredictor:
    """Advance consistent movement by at most one captured frame, never across a loss."""

    def __init__(self, *, stale_s: float | None = None):
        self.stale_s = stale_s
        self.history: deque[Intent] = deque(maxlen=3)

    def update(self, intent: Intent, published_t: float, *, recentre: bool = False) -> Intent:
        if recentre:
            self.history.clear()
        finite = all(math.isfinite(v) for v in (intent.x, intent.y, intent.t_s, published_t))
        if not intent.valid or not finite:
            self.history.clear()
            stamp = intent.t_s if math.isfinite(intent.t_s) else 0.0
            return Intent(0.0, 0.0, False, stamp)
        measured = _bounded(intent)
        if self.history and measured.t_s <= self.history[-1].t_s:
            self.history.clear()
            return measured
        if (
            self.history
            and self.stale_s is not None
            and measured.t_s - self.history[-1].t_s > self.stale_s
        ):
            self.history.clear()
        self.history.append(measured)
        if len(self.history) < 3:
            return measured
        first, middle, last = self.history
        dt0, dt1 = middle.t_s - first.t_s, last.t_s - middle.t_s
        if not (math.isfinite(dt0) and math.isfinite(dt1)):
            self.history.clear()
            return measured
        horizon = min(max(0.0, published_t - last.t_s), dt0, dt1)

        def advance(axis: str) -> float:
            v0 = (getattr(middle, axis) - getattr(first, axis)) / dt0
            v1 = (getattr(last, axis) - getattr(middle, axis)) / dt1
            agrees = v0 > 0 and v1 > 0 or v0 < 0 and v1 < 0
            finite = math.isfinite(v0) and math.isfinite(v1)
            speed = math.copysign(min(abs(v0), abs(v1)), v1) if agrees and finite else 0.0
            return getattr(last, axis) + speed * horizon

        return _bounded(Intent(advance("x"), advance("y"), True, last.t_s))
