import asyncio
import time

import uvicorn

from src import safety
from src.bus import Bus
from src.contracts import BallState, RunLogRecord
from src.control.pulse_map import tilt_to_pulse
from src.inputs.hand import HandInput
from src.inputs.keyboard import KeyboardInput
from src.io.camera import CameraSource, open_camera
from src.io.ports import Frame, FrameSource, InputSource
from src.io.settings import env, load
from src.modes import MODES
from src.sim.ball import roll
from src.sim.plate import SimActuator
from src.web.server import make_app


class Loop:
    def __init__(
        self, cfg: dict, hand: InputSource | None = None, frames: FrameSource | None = None
    ):
        self.cfg = cfg
        self.bus = Bus()
        self.keys = KeyboardInput()
        self.hand = hand
        self.frames = frames
        self.frame: Frame | None = None
        self.safety = safety.SafetyState()
        self.actuator = SimActuator(cfg["rig"], cfg["control"])
        self.mode = cfg["control"]["mode"]
        self.ball: BallState | None = None
        self.last: RunLogRecord | None = None

    def arm(self) -> None:
        self.safety.arm()

    def kill(self) -> None:
        self.safety.kill()

    def tick(self, now: float, dt: float) -> RunLogRecord:
        if self.frames is not None:
            frame = self.frames.read()
            if frame is not None:
                self.frame = frame
        rig = self.cfg["rig"]
        ball = self.ball or BallState(0.0, 0.0, 0.0, 0.0, valid=True, t_s=now)
        intent = self.keys.latest()
        if self.hand is not None:
            hand = self.hand.latest()
            if hand.valid and now - hand.t_s <= rig["stale_intent_s"]:
                intent = hand
        stale = now - intent.t_s > rig["stale_intent_s"]

        plate_cmd = MODES[self.mode](ball, intent, self.cfg)
        target = tilt_to_pulse(plate_cmd, rig)
        sent = safety.step(self.safety, target, stale, dt, rig)
        plate_state = self.actuator.send(sent, now)
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

    def snapshot(self) -> dict:
        snap = self.last.to_dict() if self.last else {}
        snap["killing"] = self.safety.killing
        return snap


async def run(loop: Loop) -> None:
    period = 1 / loop.cfg["control"]["sim_tick_hz"]
    last = time.monotonic()
    while True:
        await asyncio.sleep(period)
        now = time.monotonic()
        loop.tick(now, now - last)
        last = now


def main() -> None:
    cfg = load()
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
    loop = Loop(cfg, hand, frames)
    port = int(env("PORT", "8000"))
    server = uvicorn.Server(
        uvicorn.Config(make_app(loop), host="127.0.0.1", port=port, log_level="warning")
    )

    async def go():
        ticker = asyncio.create_task(run(loop))
        print(f"http://127.0.0.1:{port}")
        await server.serve()
        ticker.cancel()

    asyncio.run(go())


if __name__ == "__main__":
    main()
