from src.contracts import PlateState, ServoCommand
from src.control.pulse_map import pulse_to_tilt


class SimActuator:
    """Servos follow the command instantly for now. Real servo lag comes later."""

    def __init__(self, rig: dict):
        self.rig = rig
        self.state = PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0)

    def send(self, cmd: ServoCommand | None) -> PlateState:
        if cmd is not None:
            tilt = pulse_to_tilt(cmd, self.rig)
            self.state = PlateState(
                sent_x_deg=tilt.tilt_x_deg,
                sent_y_deg=tilt.tilt_y_deg,
                est_x_deg=tilt.tilt_x_deg,
                est_y_deg=tilt.tilt_y_deg,
            )
        return self.state
