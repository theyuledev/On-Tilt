from src.contracts import PlateCommand, ServoCommand


def center(rig: dict) -> tuple[float, float]:
    return (
        rig["center_x_us"] + rig["trim_x_us"],
        rig["center_y_us"] + rig["trim_y_us"],
    )


def tilt_to_pulse(cmd: PlateCommand, rig: dict) -> ServoCommand:
    cx, cy = center(rig)
    return ServoCommand(
        x_us=round(cx + rig["us_per_deg_x"] * cmd.tilt_x_deg),
        y_us=round(cy + rig["us_per_deg_y"] * cmd.tilt_y_deg),
    )


def pulse_to_tilt(cmd: ServoCommand, rig: dict) -> PlateCommand:
    cx, cy = center(rig)
    return PlateCommand(
        tilt_x_deg=(cmd.x_us - cx) / rig["us_per_deg_x"],
        tilt_y_deg=(cmd.y_us - cy) / rig["us_per_deg_y"],
    )
