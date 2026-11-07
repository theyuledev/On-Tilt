from src.contracts import BallState, Intent, PlateCommand


def hand(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not intent.valid:
        return PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0)
    inp = cfg["input"]

    def shape(v: float) -> float:
        return 0.0 if abs(v) < inp["dead_zone"] else v

    gain = inp["intent_gain_deg"]
    return PlateCommand(tilt_x_deg=gain * shape(intent.x), tilt_y_deg=gain * shape(intent.y))


def balance(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not ball.valid:
        return PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0)
    c = cfg["control"]

    def pd(pos: float, vel: float) -> float:
        d = min(max(c["kd_deg_s_mm"] * vel, -c["d_max_deg"]), c["d_max_deg"])
        return -(c["kp_deg_mm"] * pos + d)

    return PlateCommand(pd(ball.x_mm, ball.vx_mm_s), pd(ball.y_mm, ball.vy_mm_s))


MODES = {"hand": hand, "balance": balance}
