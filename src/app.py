import asyncio
import time

import uvicorn

from src import safety
from src.bus import Bus
from src.contracts import BallState, PlateState, RunLogRecord, ServoCommand
from src.control.pulse_map import tilt_to_pulse
from src.inputs.hand import HandInput
from src.inputs.keyboard import KeyboardInput
from src.io.camera import CameraSource, open_camera
from src.io.ports import Frame, FrameSource, InputSource, TiltActuator
from src.io.serial_link import LinkError, SerialActuator, open_board
from src.io.settings import env, load
from src.modes import MODES
from src.sim.ball import roll
from src.sim.plate import SimActuator
from src.web.server import make_app

LEVEL = PlateState(sent_x_deg=0.0, sent_y_deg=0.0, est_x_deg=0.0, est_y_deg=0.0)


class Loop:
    def __init__(
        self,
        cfg: dict,
        hand: InputSource | None = None,
        frames: FrameSource | None = None,
        actuator: TiltActuator | None = None,
    ):
        self.cfg = cfg
        self.bus = Bus()
        self.keys = KeyboardInput()
        self.hand = hand
        self.frames = frames
        self.frame: Frame | None = None
        self.safety = safety.SafetyState()
        self.actuator = actuator or SimActuator(cfg["rig"], cfg["control"])
        self.board_lost = False
        self.stale = False
        self.mode = cfg["control"]["mode"]
        self.ball: BallState | None = None
        self.last: RunLogRecord | None = None

    def arm(self) -> None:
        if not self.board_lost:
            self.safety.arm()

    def kill(self) -> None:
        self.safety.kill()

    def tick(self, now: float, dt: float) -> RunLogRecord:
        if self.frames is not None:
            frame = self.frames.read()
            if frame is not None:
                self.frame = frame
        rig = self.cfg["rig"]
        sim = isinstance(self.actuator, SimActuator)
        ball = self.ball or BallState(0.0, 0.0, 0.0, 0.0, valid=sim, t_s=now)
        intent = self.keys.latest()
        if self.hand is not None:
            hand = self.hand.latest()
            if hand.valid and now - hand.t_s <= rig["stale_intent_s"]:
                intent = hand
        stale = self.stale = now - intent.t_s > rig["stale_intent_s"]

        plate_cmd = MODES[self.mode](ball, intent, self.cfg)
        target = tilt_to_pulse(plate_cmd, rig)
        sent, plate_state = self._drive(target, stale, dt, now)
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

    def _drive(
        self, target: ServoCommand, stale: bool, dt: float, now: float
    ) -> tuple[ServoCommand | None, PlateState]:
        if self.board_lost:
            return None, LEVEL
        sent = safety.step(self.safety, target, stale, dt, self.cfg["rig"])
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
        snap["hand_seen"] = self.hand is not None and self.hand.seen
        snap["hand_calibrating"] = self.hand is not None and self.hand.calibrating
        return snap


async def run(loop: Loop) -> None:
    period = 1 / loop.cfg["control"]["sim_tick_hz"]
    last = time.monotonic()
    while True:
        await asyncio.sleep(period)
        now = time.monotonic()
        loop.tick(now, now - last)
        last = now


def stop_on_error(ticker: asyncio.Task, server) -> None:
    # a dead loop must not leave the page looking alive
    if not ticker.cancelled() and ticker.exception() is not None:
        print(f"loop died: {ticker.exception()!r}")
        server.should_exit = True


def main() -> None:
    cfg = load()
    mode = cfg["control"]["mode"]
    if mode not in MODES:
        raise SystemExit(f"control.mode {mode!r} is not one of {sorted(MODES)}")
    try:
        hand = HandInput(cfg["input"]["hand_port"])
    except OSError as e:
        print(f"hand input off: {e}")
        hand = None
    try:
        frames = CameraSource(open_camera(cfg["rig"]))
    except RuntimeError as e:
        print(f"camera off: {e}")
        frames = None
    actuator = None
    if cfg["rig"]["actuator"] == "real":
        actuator = SerialActuator(open_board(cfg["rig"]), cfg["rig"])
    loop = Loop(cfg, hand, frames, actuator)
    port = int(env("PORT", "8000"))
    server = uvicorn.Server(
        uvicorn.Config(make_app(loop), host="127.0.0.1", port=port, log_level="warning")
    )

    async def go():
        ticker = asyncio.create_task(run(loop))
        ticker.add_done_callback(lambda t: stop_on_error(t, server))
        print(f"http://127.0.0.1:{port}")
        await server.serve()
        ticker.cancel()

    asyncio.run(go())


if __name__ == "__main__":
    main()
