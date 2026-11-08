import math
from dataclasses import replace

import numpy as np

from src.contracts import BallState, ServoCommand
from src.sim.ball import roll
from src.sim.plate import SimActuator
from src.vision.pipeline import Observation
from src.vision.tracker import BallPx

# a made up rig for the --sim runs: what autocal should find
TRUTH = {
    "yaw_deg": 37.0,
    "mirror": False,
    "s_ball": 2.6,
    "centre_px": (700.0, 380.0),
    "trim_x_us": 15.0,
    "trim_y_us": -25.0,
    "us_per_deg_x": 57.0,
    "us_per_deg_y": 57.0,
    "friction_mm_s2": 15.0,
    "noise_px": 0.36,
    "r_short_px": 0.5,  # the contour area reads the radius a bit small
    "servo_dead_time_s": 0.046,
}


class SimRig:
    """Servos, plate and ball on a fake clock, seen through a flat camera map."""

    def __init__(self, cfg: dict, truth: dict = TRUTH, seed: int = 0):
        self.truth = truth
        self.rig = {**cfg["rig"], **{k: truth[k] for k in ("trim_x_us", "trim_y_us")}}
        self.rig.update({k: truth[k] for k in ("us_per_deg_x", "us_per_deg_y")})
        self.control = {**cfg["control"], "servo_dead_time_s": truth["servo_dead_time_s"]}
        self.servo = SimActuator(self.rig, self.control)
        self.ball = BallState(0.0, 0.0, 0.0, 0.0, True, 0.0)
        self.t = 0.0
        self.visible = True
        self.rng = np.random.default_rng(seed)

    def clock(self) -> float:
        return self.t

    def send(self, x_us: int, y_us: int) -> None:
        self.servo.send(ServoCommand(x_us, y_us), self.t)

    def advance(self, dt: float) -> None:
        self.t += dt
        plate = self.servo.send(None, self.t)
        b = roll(self.ball, plate, self.t, self.rig, self.control)
        speed, drop = math.hypot(b.vx_mm_s, b.vy_mm_s), self.truth["friction_mm_s2"] * dt
        keep = max(speed - drop, 0.0) / speed if speed else 0.0
        self.ball = replace(b, vx_mm_s=b.vx_mm_s * keep, vy_mm_s=b.vy_mm_s * keep)

    def close(self) -> None:
        pass

    def latest(self) -> Observation:
        tr, b = self.truth, self.ball
        y = -b.y_mm if tr["mirror"] else b.y_mm
        a = math.radians(-tr["yaw_deg"])
        u, v = b.x_mm * math.cos(a) - y * math.sin(a), b.x_mm * math.sin(a) + y * math.cos(a)
        nx, ny = self.rng.normal(0.0, tr["noise_px"], 2)
        s, (cx, cy) = tr["s_ball"], tr["centre_px"]
        r = s * self.rig["ball_d_mm"] / 2 - tr["r_short_px"]
        px = BallPx(cx + s * u + nx, cy - s * v + ny, r, self.visible, self.t)
        return Observation(b, px, math.nan, math.nan, False, self.t, 0.0)


TIMEOUT_S = 0.25  # firmware/tilt TIMEOUT_MS: silent this long while armed and the board disarms


class SimLink:
    """The board's A/P/D/L/S lines, driving a SimRig."""

    def __init__(self, world: SimRig):
        self.world = world
        self.armed = False
        self.limits: list[int] | None = None
        self.sent: list[str] = []
        self.t_last = world.t

    def raw(self, line: str) -> str:
        self.sent.append(line)
        if self.armed and self.world.t - self.t_last > TIMEOUT_S:
            self.armed = False
        self.t_last = self.world.t
        cmd, *args = line.split()
        args = [int(a) for a in args]
        if cmd == "L":
            if self.armed:
                return "ERR L armed"
            self.limits = args
            return "OK L"
        if cmd == "A":
            if self.limits is None:
                return "ERR A nolimits"
            self.armed = True
            self.world.send(*self.limits[4:])
            return "OK A"
        if cmd == "P":
            if not self.armed:
                return "ERR P disarmed"
            lo_x, hi_x, lo_y, hi_y = self.limits[:4]
            x, y = min(max(args[0], lo_x), hi_x), min(max(args[1], lo_y), hi_y)
            self.world.send(x, y)
            return f"OK P {x} {y}"
        if cmd == "D":
            self.armed = False
            return "OK D"
        if cmd == "S":
            return f"OK S {int(self.armed)} {int(self.limits is not None)} 0 sim"
        return "ERR ?"
