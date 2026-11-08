from src.contracts import BallState, Intent, PlateCommand
from src.control.pid import Integral, integrate, pid, smooth
from src.inputs.shaping import shape
from src.modes.assist import goal_mm
from src.modes.path import clip, target

LEVEL = PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0)


def hand(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not intent.valid:
        return LEVEL
    return PlateCommand(*shape(intent.x, intent.y, cfg["input"]))


def pd_toward(ball: BallState, goal: tuple[float, float], cfg: dict) -> PlateCommand:
    """Tilt that pulls the ball to goal and brakes its motion."""
    c = cfg["control"]
    return PlateCommand(
        pid(ball.x_mm - goal[0], ball.vx_mm_s, 0.0, c),
        pid(ball.y_mm - goal[1], ball.vy_mm_s, 0.0, c),
    )


class Balance:
    def __init__(self, cfg: dict):
        self.acc = Integral()
        self.vel: tuple[float, float] | None = None

    def __call__(self, ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
        c = cfg["control"]
        dt = 0.0 if self.acc.t_s is None else max(ball.t_s - self.acc.t_s, 0.0)
        self.acc = integrate(self.acc, ball, c)
        if not ball.valid:
            self.vel = None
            return LEVEL
        raw = (ball.vx_mm_s, ball.vy_mm_s)
        prev = self.vel or raw
        vx, vy = (smooth(p, v, dt, c["vel_ema_s"]) for p, v in zip(prev, raw, strict=True))
        self.vel = (vx, vy)
        return PlateCommand(
            tilt_x_deg=pid(ball.x_mm, vx, self.acc.x, c),
            tilt_y_deg=pid(ball.y_mm, vy, self.acc.y, c),
        )


def assist(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not ball.valid:
        return LEVEL
    c = cfg["control"]
    goal = goal_mm(intent, cfg["rig"], c["assist_margin_mm"])
    # a far goal is chased like a path point, else the pd overshoots into the rim
    return pd_toward(ball, clip((ball.x_mm, ball.y_mm), goal, c["path_lookahead_mm"]), cfg)


class PathFollower:
    """Chases the point path_lookahead_mm ahead along control.path_mm, then holds the end."""

    def __init__(self, cfg: dict):
        self.path = [(p[0], p[1]) for p in cfg["control"]["path_mm"]]
        if not self.path:
            raise ValueError("control.path_mm needs at least one point")
        self.seg = 0

    def __call__(self, ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
        if not ball.valid:
            return LEVEL
        c = cfg["control"]
        at = (ball.x_mm, ball.y_mm)
        ahead, capture = c["path_lookahead_mm"], c["path_capture_mm"]
        self.seg, goal = target(self.path, self.seg, at, ahead, capture)
        return pd_toward(ball, goal, cfg)


MODES = {"hand": hand, "balance": Balance, "path": PathFollower, "assist": assist}
# the modes that steer on the tracked ball rather than a player
CLOSED_LOOP = ("balance", "path", "assist")


def make_mode(name: str, cfg: dict):
    """The callable for one run of a mode; modes that keep state are classes built from cfg."""
    mode = MODES[name]
    return mode(cfg) if isinstance(mode, type) else mode
