from src import safety
from src.bus import Bus
from src.contracts import BallState, Intent, PlateCommand, PlateState, RunLogRecord, ServoCommand
from src.control.deadband import invert, start
from src.control.pulse_map import tilt_to_pulse
from src.estimate import BallEstimator
from src.inputs.keyboard import KeyboardInput
from src.io.ports import Frame, FrameSource, InputSource, TiltActuator
from src.io.serial_link import LinkError
from src.modes import CLOSED_LOOP, arm_block, fade_in, fade_start, make_mode
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
        game=None,
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
        self.holding = self.stale = self.page_stale = False
        self.armed_s, self.page_t = 0.0, None
        self.hand_seen = self.hand_calibrating = False
        self.mode = cfg["control"]["mode"]
        self.ball: BallState | None = None
        self.ball_px: BallPx | None = None
        self.seen_t: float | None = None
        self.est = BallEstimator(cfg)
        self.exposure = exposure
        self.last: RunLogRecord | None = None
        self.game = game

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, name: str) -> None:
        # picking a mode by name builds its callable; a path starts over at its first segment
        self._mode, self.step = name, make_mode(name, self.cfg)

    @property
    def arm_blocked(self) -> str | None:
        return arm_block(self.exposure, self.mode)

    def arm(self) -> None:
        if not self.board_lost and not self.arm_blocked:
            self.armed_s = fade_start(self.armed_s, self.holding, self.safety.armed)
            self.step = make_mode(self.mode, self.cfg)
            self.safety.arm()
            self.holding, self.page_t = False, None

    def hold(self) -> None:
        # level between runs so the next arm starts from level, not the sag; never arms
        self.holding = self.holding or (self.safety.armed and not self.safety.killing)

    def kill(self) -> None:
        self.holding = False
        self.safety.kill()

    def page_seen(self, now: float) -> None:
        self.page_t = now

    def tick(self, now: float, dt: float) -> RunLogRecord:
        if self.frames is not None:
            self._see()
        rig = self.cfg["rig"]
        sim = isinstance(self.actuator, SimActuator) and self.frames is None
        ball = self.ball or BallState(0.0, 0.0, 0.0, 0.0, valid=sim, t_s=now)
        ball = safety.aged(ball, now, rig)
        intent, stale = self._intent(now)
        if self.game is not None:
            want = self.game.mode() or self.cfg["control"]["mode"]
            if want != self.mode:
                self.mode = want
                if self.safety.armed and self.arm_blocked:
                    self.kill()

        if not self.safety.armed or self.safety.killing or stale:
            self.step = make_mode(self.mode, self.cfg)  # fresh mode state, no kick on arm
        plate_cmd = self.step(ball, intent, self.cfg)
        self.armed_s += dt
        target = tilt_to_pulse(fade_in(plate_cmd, self.holding, self.armed_s, rig), rig)
        sent, plate_state = self._drive(target, stale, dt, now)
        tilt = (plate_state.sent_x_deg, plate_state.sent_y_deg) if sent is not None else (0.0, 0.0)
        self._watch("command", now, PlateCommand(*tilt))
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

    def _intent(self, now: float) -> tuple[Intent, bool]:
        rig, quiet_s = self.cfg["rig"], self.cfg["game"]["heartbeat_stale_s"]
        intent = self.keys.latest()
        if self.hand is not None:
            hand = self.hand.latest()
            fresh = now - hand.t_s <= rig["stale_intent_s"]
            if hand.valid and fresh:
                intent = hand
            # a dead hand process must not leave the page saying the hand is live
            self.hand_seen = fresh and self.hand.seen
            self.hand_calibrating = fresh and self.hand.calibrating
        self.page_stale = safety.page_quiet(self.safety.armed, self.page_t, now, quiet_s)
        self.stale = now - intent.t_s > rig["stale_intent_s"] or self.page_stale
        return intent, self.stale

    def _see(self) -> None:
        frame = self.frames.read()
        if frame is not None:
            self.frame = frame
        obs = self.frames.latest()
        if obs is None or obs.t_read == self.seen_t:
            return
        self.seen_t = obs.t_read
        self.ball_px, self.ball = obs.ball_px, obs.ball
        if obs.ball.valid:
            self._watch("update", obs.t_read, obs.ball.x_mm, obs.ball.y_mm)
        else:
            self._watch("miss", obs.t_read)

    def _watch(self, call: str, *args):
        # the estimator only watches, so a fault in it switches it off and the plate carries on
        try:
            return getattr(self.est, call)(*args) if self.est is not None else None
        except Exception as e:
            print(f"estimator off: {e!r}")
            self.est = None

    def _drive(
        self, target: ServoCommand, stale: bool, dt: float, now: float
    ) -> tuple[ServoCommand | None, PlateState]:
        if self.board_lost:
            return None, LEVEL
        sent = safety.step(self.safety, target, stale, dt, self.cfg["rig"])
        if self.cfg["control"]["inverter_on"]:
            # only the closed loop modes get the dead band help, a player closes the loop himself
            idle = not self.safety.armed or self.safety.killing or stale or self.holding
            off = idle or self.mode not in CLOSED_LOOP
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
        snap["holding"] = self.holding and self.safety.armed
        snap["stale"] = self.stale
        snap["page_stale"] = self.page_stale
        snap["board_lost"] = self.board_lost
        snap["camera_lost"] = self.camera_lost
        px = self.ball_px
        snap["ball_px"] = [px.x_px, px.y_px, px.r_px] if px is not None and px.valid else None
        snap["hand_seen"] = self.hand_seen
        snap["hand_calibrating"] = self.hand_calibrating
        snap["hand_progress"] = self.hand.progress if self.hand is not None else 0.0
        e = self._watch("predict", self.last.t_s) if self.last else None
        snap["est"] = [e.x_mm, e.y_mm, e.vx_mm_s, e.vy_mm_s] if e is not None and e.valid else None
        snap["exposure"] = self.exposure
        snap["arm_blocked"] = self.arm_blocked
        if self.game is not None:
            snap["race"] = self.game.view()
        return snap
