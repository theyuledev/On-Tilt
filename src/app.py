import asyncio
import time

import uvicorn

from src import safety
from src.bus import Bus
from src.contracts import BallState, RunLogRecord
from src.control.pulse_map import tilt_to_pulse
from src.inputs.keyboard import KeyboardInput
from src.io.settings import env, load
from src.modes import MODES
from src.sim.plate import SimActuator
from src.web.server import make_app


class Loop:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.bus = Bus()
        self.keys = KeyboardInput()
        self.safety = safety.SafetyState()
        self.actuator = SimActuator(cfg["rig"], cfg["control"])
        self.mode = cfg["control"]["mode"]
        self.last: RunLogRecord | None = None

    def arm(self) -> None:
        self.safety.arm()

    def kill(self) -> None:
        self.safety.kill()

    def tick(self, now: float, dt: float) -> RunLogRecord:
        rig = self.cfg["rig"]
        ball = BallState(x_mm=0.0, y_mm=0.0, vx_mm_s=0.0, vy_mm_s=0.0, valid=False, t_s=now)
        intent = self.keys.latest()
        stale = now - intent.t_s > rig["stale_intent_s"]

        plate_cmd = MODES[self.mode](ball, intent, self.cfg["input"])
        target = tilt_to_pulse(plate_cmd, rig)
        sent = safety.step(self.safety, target, stale, dt, rig)
        plate_state = self.actuator.send(sent, now)

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
    loop = Loop(load())
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
