from dataclasses import replace

from src import safety
from src.bus import Bus
from src.contracts import BallState, Intent, PlateCommand, PlateState, RunLogRecord, ServoCommand
from src.control.deadband import invert, start
from src.control.pulse_map import tilt_to_pulse
from src.estimate import BallEstimator
from src.inputs.keyboard import KeyboardInput
from src.io.ports import Frame, FrameSource, InputSource, TiltActuator
from src.io.serial_link import LinkError
from src.modes import make_mode
from src.sim.ball import roll
from src.sim.plate import SimActuator
from src.vision.tracker import BallPx

LEVEL = PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0)


class Loop:
    def __init__(
        self,
        cfg: dict,
        hand: InputSource | None = None,
        frames: FrameSource | None = None,
        actuator: TiltActuator | None = None,
        exposure: str | None = None,
    ):
        self.cfg = cfg
        self.bus = Bus()
        self.keys = KeyboardInput(cfg["input"])
        self.hand = hand
        self.frames = frames
        self.frame: Frame | None = None
        self.safety = safety.SafetyState()
        self.band = start(cfg["rig"])
        self.actuator = actuator or SimActuator(cfg["rig"], cfg["control"])
        self.board_lost = False
        self.stale = False
        self.hand_seen = self.hand_calibrating = False
        self.mode = cfg["control"]["mode"]
        self.ball: BallState | None = None
        self.ball_px: BallPx | None = None
        self.seen_t: float | None = None
        # watches only: nothing it predicts reaches a mode yet
        self.est = BallEstimator(cfg)
        self.exposure = exposure
        self.last: RunLogRecord | None = None

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, name: str) -> None:
        # picking a mode by name builds its callable; a path starts over at its first segment
        self._mode, self.step = name, make_mode(name, self.cfg)

    @property
    def arm_blocked(self) -> str | None:
        # a wrong exposure smears the ball, so the modes steering on it wait for a fix
        e = self.exposure or ""
        closed = self.mode in ("balance", "path", "assist")
        return e if closed and e and not e.startswith("exposure unverified") else None

    def arm(self) -> None:
        if not self.board_lost and not self.arm_blocked:
            self.step = make_mode(self.mode, self.cfg)
            self.safety.arm()

    def kill(self) -> None:
        self.safety.kill()

    def tick(self, now: float, dt: float) -> RunLogRecord:
        if self.frames is not None:
            self._see()
        rig = self.cfg["rig"]
        sim = isinstance(self.actuator, SimActuator) and self.frames is None
        ball = self.ball or BallState(0.0, 0.0, 0.0, 0.0, valid=sim, t_s=now)
        if now - ball.t_s > rig["stale_intent_s"]:
            # an old frame says nothing about where the ball is now
            ball = replace(ball, valid=False)
        intent = self._intent(now)
        stale = self.stale

        if not self.safety.armed or self.safety.killing or stale:
            self.step = make_mode(self.mode, self.cfg)  # fresh mode state, no kick on arm
        plate_cmd = self.step(ball, intent, self.cfg)
        target = tilt_to_pulse(plate_cmd, rig)
        sent, plate_state = self._drive(target, stale, dt, now)
        tilt = (plate_state.sent_x_deg, plate_state.sent_y_deg) if sent is not None else (0.0, 0.0)
        self.est.command(now, PlateCommand(*tilt))
        if sim:
            self.ball = roll(ball, plate_state, now, rig, self.cfg["control"])

        rec = RunLogRecord(
            t_s=now,
            mode=self.mode,
            armed=sent is not None,
            ball=ball,
            intent=intent,
            plate_cmd=plate_cmd,
            servo_cmd=sent or target,
            plate_state=plate_state,
        )
        self.last = rec
        self.bus.publish("tick", rec)
        return rec

    def _intent(self, now: float) -> Intent:
        rig = self.cfg["rig"]
        intent = self.keys.latest()
        if self.hand is not None:
            hand = self.hand.latest()
            fresh = now - hand.t_s <= rig["stale_intent_s"]
            if hand.valid and fresh:
                intent = hand
            # a dead hand process must not leave the page saying the hand is live
            self.hand_seen = fresh and self.hand.seen
            self.hand_calibrating = fresh and self.hand.calibrating
        self.stale = now - intent.t_s > rig["stale_intent_s"]
        return intent

    def _see(self) -> None:
        frame = self.frames.read()
        if frame is not None:
            self.frame = frame
        # the pipeline runs on the camera thread, so take its newest result once
        obs = self.frames.latest()
        if obs is None or obs.t_read == self.seen_t:
            return
        self.seen_t = obs.t_read
        self.ball_px, self.ball = obs.ball_px, obs.ball
        if obs.ball.valid:
            self.est.update(obs.t_read, obs.ball.x_mm, obs.ball.y_mm)
        else:
            self.est.miss(obs.t_read)

    def _drive(
        self, target: ServoCommand, stale: bool, dt: float, now: float
    ) -> tuple[ServoCommand | None, PlateState]:
        if self.board_lost:
            return None, LEVEL
        sent = safety.step(self.safety, target, stale, dt, self.cfg["rig"])
        if self.cfg["control"]["inverter_on"]:
            # only the closed loop modes get the dead band help, a player closes the loop himself
            off = (
                not self.safety.armed
                or self.safety.killing
                or stale
                or self.mode not in ("balance", "path", "assist")
            )
            self.band, sent = invert(self.band, sent, off, self.cfg)
        try:
            return sent, self.actuator.send(sent, now)
        except LinkError as e:
            # board refused or went quiet and levels itself, so disarm and let the page re-arm
            print(f"board disarmed: {e}")
            self.actuator.armed = False
            self.safety = safety.SafetyState()
            return None, LEVEL
        except (OSError, ValueError) as e:
            # port gone or garbled, nothing sane to send any more
            print(f"board lost: {e}")
            self.board_lost = True
            self.safety = safety.SafetyState()
            return None, LEVEL

    @property
    def camera_lost(self) -> bool:
        # the grab thread stops on a failed read; clip sources have no thread
        return self.frames is not None and not getattr(self.frames, "running", True)

    def snapshot(self) -> dict:
        snap = self.last.to_dict() if self.last else {}
        snap["killing"] = self.safety.killing
        snap["stale"] = self.stale
        snap["board_lost"] = self.board_lost
        snap["camera_lost"] = self.camera_lost
        px = self.ball_px
        snap["ball_px"] = [px.x_px, px.y_px, px.r_px] if px is not None and px.valid else None
        snap["hand_seen"] = self.hand_seen
        snap["hand_calibrating"] = self.hand_calibrating
        snap["hand_progress"] = self.hand.progress if self.hand is not None else 0.0
        e = self.est.predict(self.last.t_s) if self.last else None
        snap["est"] = [e.x_mm, e.y_mm, e.vx_mm_s, e.vy_mm_s] if e is not None and e.valid else None
        snap["exposure"] = self.exposure
        snap["arm_blocked"] = self.arm_blocked
        return snap
