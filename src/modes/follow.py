"""Path mode for a path with two ends: from the end by the ball to the other, then hold it."""

import math
from dataclasses import replace

from src.contracts import BallState, Intent, PlateCommand
from src.control import ROLL
from src.control.pid import Integral, integrate, smooth
from src.modes.path import (
    from_nearest_end,
    heading,
    lengths,
    progress,
    seg_len,
    sight,
    stretch,
    tangent,
    tidy,
    turned,
    walk,
)
from src.modes.spot import inside


class OpenPath:
    """Waits for a ball, joins the end nearer to it, follows the line and finishes with the
    balance law about the last point. A ball at rest always gets the pull of the full
    lookahead, the tilt that started it; a moving one gets less where the line bends ahead."""

    def __init__(self, path: list, cfg: dict, hold, pull):
        self.path = tidy([inside(p, cfg) for p in path])
        self.new_hold, self.pull = hold, pull
        self.hold = hold()
        self.phase, self.s, self.done = "wait", 0.0, False
        self.vel: tuple[float, float] | None = None
        self.t_s = 0.0
        self.acc, self.chase = Integral(), None

    def resume(self) -> None:
        """The same run goes on after a gap in the heartbeat, without what was wound up."""
        self.hold, self.acc = self.new_hold(), Integral()

    def __call__(self, ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
        if self.phase == "finish":
            return self._finish(ball, cfg)
        if not ball.valid:
            # an old speed or lean says nothing after a blind stretch
            self.vel, self.acc = None, Integral()
            return PlateCommand(0.0, 0.0)
        c, at = cfg["control"], (ball.x_mm, ball.y_mm)
        if self.phase == "wait":
            self.path = from_nearest_end(self.path, at)
            self.cum = lengths(self.path)
            self.phase = "join"
        if self.phase == "join" and seg_len(at, self.path[0]) <= c["path_capture_mm"]:
            self.phase = "follow"
        if self.phase == "join":
            self.chase = self.path[0]
            return self.pull(
                self._slowed(ball, c), stretch(at, self.chase, c["path_lookahead_mm"]), cfg
            )
        self._advance(at, c)
        # balance takes the last stretch once the ball is near enough for its integral to wind
        if (
            self.s + c["path_lookahead_mm"] >= self.cum[-1]
            and seg_len(at, self.path[-1]) < c["i_err_max_mm"]
        ):
            self.phase, self.hold.vel, self.chase = "finish", self.vel, self.path[-1]
            return self._finish(ball, cfg)
        cmd = self.pull(self._slowed(ball, c), self._aim(at, c), cfg)
        # a lean holds the ball off the line: wind against how far off it is, across the line only
        here, t = walk(self.path, 0, 0.0, self.s), tangent(self.path, self.cum, self.s)
        off = (at[1] - here[1]) * math.cos(t) - (at[0] - here[0]) * math.sin(t)
        self.acc = integrate(
            self.acc, replace(ball, x_mm=-off * math.sin(t), y_mm=off * math.cos(t)), c
        )
        return PlateCommand(cmd.tilt_x_deg - self.acc.x, cmd.tilt_y_deg - self.acc.y)

    def _ahead(self, c: dict) -> float:
        """How far the line runs on straight enough to aim along: at most the lookahead, at least
        path_capture_mm so a sharp corner is still looked round."""
        seen = sight(self.path, self.cum, self.s, c["path_lookahead_mm"], c["path_sight_tol_mm"])
        return max(seen, min(c["path_capture_mm"], self.cum[-1] - self.s))

    def _advance(self, at: tuple[float, float], c: dict) -> None:
        """The place moves on along the straight part ahead. Where the line bends within the
        lookahead it moves only while the ball is within path_gate_mm of the line there, so a ball
        that cut across is brought back; on a straight that would hold a swing that dies out."""
        far = self._ahead(c)
        new = progress(self.path, self.cum, at, self.s, far)
        if (
            far >= c["path_lookahead_mm"]
            or seg_len(at, walk(self.path, 0, 0.0, new)) <= c["path_gate_mm"]
        ):
            self.s = new

    def _aim(self, at: tuple[float, float], c: dict) -> tuple[float, float]:
        """The goal of the pull. Its length shrinks from the lookahead to the straight part ahead
        as the ball nears path_bend_mm_s, which slows it before a bend. It points the way the line
        turns over the distance this pull needs to turn a ball going this fast along it, turned
        back toward the line as seen from the end of the straight part."""
        ahead, far = c["path_lookahead_mm"], self._ahead(c)
        v, t = math.hypot(*self.vel), tangent(self.path, self.cum, self.s)
        bend = c["path_bend_mm_s"]
        reach = ahead - (ahead - far) * (min(v / bend, 1.0) if bend > 0 else 1.0)
        push = ROLL * c["sim_g_mm_s2"] * math.pi / 180 * c["kp_deg_mm"] * reach  # mm/s^2
        # only the speed along the line: a sideways swing would otherwise swing the turn too
        along = max(self.vel[0] * math.cos(t) + self.vel[1] * math.sin(t), 0.0)
        turn = min(2 * along**2 / push, far) if push > 0 else far
        here, near, self.chase = (walk(self.path, 0, 0.0, self.s + d) for d in (0.0, turn, far))
        if here == self.chase:
            return stretch(at, self.chase, reach)
        way = heading(here, near) if near != here else t
        way += turned(heading(here, self.chase), heading(at, self.chase))
        return at[0] + reach * math.cos(way), at[1] + reach * math.sin(way)

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
