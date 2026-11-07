from src.contracts import BallState, Intent, PlateCommand


def hand(ball: BallState, intent: Intent, cfg: dict) -> PlateCommand:
    if not intent.valid:
        return PlateCommand(tilt_x_deg=0.0, tilt_y_deg=0.0)

    def shape(v: float) -> float:
        return 0.0 if abs(v) < cfg["dead_zone"] else v

    gain = cfg["intent_gain_deg"]
    return PlateCommand(tilt_x_deg=gain * shape(intent.x), tilt_y_deg=gain * shape(intent.y))


MODES = {"hand": hand}
