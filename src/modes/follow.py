"""Path mode for a path with two ends: from the end by the ball to the other, then hold it."""

from dataclasses import replace

from src.contracts import BallState, Intent, PlateCommand
from src.control.pid import smooth
from src.modes.path import from_nearest_end, lengths, progress, seg_len, stretch, tidy, walk
from src.modes.spot import inside


class OpenPath:
    """Waits for a ball, joins the end nearer to it, follows the point a lookahead further along
    and finishes with the balance law about the last point. Until then the pull always has the
    lookahead's length, so a ball that stops gets the tilt that started it."""

    def __init__(self, path: list, cfg: dict, hold, pull):
        self.path = tidy([inside(p, cfg) for p in path])
        self.new_hold, self.pull = hold, pull
        self.hold = hold()
        self.phase, self.s, self.done = "wait", 0.0, False
        self.vel: tuple[float, float] | None = None
        self.t_s = 0.0

    def resume(self) -> None:
        """The same run goes on after a gap in the heartbeat, without what the finish wound up."""
        self.hold = self.new_hold()

    def __call__(self, ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
        if self.phase == "finish":
            return self._finish(ball, cfg)
        if not ball.valid:
            self.vel = None  # an old speed says nothing after a blind stretch
            return PlateCommand(0.0, 0.0)
        c, at = cfg["control"], (ball.x_mm, ball.y_mm)
        ahead = c["path_lookahead_mm"]
        if self.phase == "wait":
            self.path = from_nearest_end(self.path, at)
            self.cum = lengths(self.path)
            self.phase = "join"
        if self.phase == "join" and seg_len(at, self.path[0]) <= c["path_capture_mm"]:
            self.phase = "follow"
        chase = self.path[0]
        if self.phase == "follow":
            self.s = progress(self.path, self.cum, at, self.s, ahead)
            # balance takes the last stretch once the ball is near enough for its integral to wind
            if self.s + ahead >= self.cum[-1] and seg_len(at, self.path[-1]) < c["i_err_max_mm"]:
                self.phase, self.hold.vel = "finish", self.vel
                return self._finish(ball, cfg)
            chase = walk(self.path, 0, 0.0, self.s + ahead)
        return self.pull(self._slowed(ball, c), stretch(at, chase, ahead), cfg)

    def _finish(self, ball: BallState, cfg: dict) -> PlateCommand:
        end = self.path[-1]
        if ball.valid and seg_len((ball.x_mm, ball.y_mm), end) <= cfg["control"]["path_capture_mm"]:
            self.done = True
        return self.hold.toward(ball, end, cfg)

    def _slowed(self, ball: BallState, c: dict) -> BallState:
        """ball with its speed smoothed over vel_ema_s, the speed balance brakes on."""
        raw = (ball.vx_mm_s, ball.vy_mm_s)
        dt = max(ball.t_s - self.t_s, 0.0) if self.vel else 0.0
        prev = self.vel or raw
        self.vel = tuple(smooth(p, v, dt, c["vel_ema_s"]) for p, v in zip(prev, raw, strict=True))
        self.t_s = ball.t_s
        return replace(ball, vx_mm_s=self.vel[0], vy_mm_s=self.vel[1])
