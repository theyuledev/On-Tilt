from src.contracts import BallState, Intent, PlateCommand
from src.modes.path import target

LEVEL = PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0)


def hand(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not intent.valid:
        return LEVEL
    inp = cfg["input"]

    def shape(v: float) -> float:
        return 0.0 if abs(v) < inp["dead_zone"] else v

    gain = inp["intent_gain_deg"]
    return PlateCommand(tilt_x_deg=gain * shape(intent.x), tilt_y_deg=gain * shape(intent.y))


def pd_toward(ball: BallState, goal: tuple[float, float], cfg: dict) -> PlateCommand:
    """Tilt that pulls the ball to goal and brakes its motion."""
    c = cfg["control"]

    def pd(err: float, vel: float) -> float:
        d = min(max(c["kd_deg_s_mm"] * vel, -c["d_max_deg"]), c["d_max_deg"])
        return -(c["kp_deg_mm"] * err + d)

    return PlateCommand(
        pd(ball.x_mm - goal[0], ball.vx_mm_s), pd(ball.y_mm - goal[1], ball.vy_mm_s)
    )


def balance(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not ball.valid:
        return LEVEL
    return pd_toward(ball, (0.0, 0.0), cfg)


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


MODES = {"hand": hand, "balance": balance, "path": PathFollower}


def make_mode(name: str, cfg: dict):
    """The callable for one run of a mode; modes that keep state are classes built from cfg."""
    mode = MODES[name]
    return mode(cfg) if isinstance(mode, type) else mode
