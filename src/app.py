import asyncio
import time

import uvicorn

from src.inputs.hand import HandInput
from src.io.camera import CameraSource, open_camera
from src.io.camera_exposure import exposure_problem
from src.io.run_log import start_run_log
from src.io.serial_link import SerialActuator, open_board
from src.io.settings import env, load
from src.loop import Loop
from src.modes import MODES
from src.vision.calib import load_lens
from src.vision.pipeline import Pipeline
from src.web.server import make_app


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


def check_exposure(rig: dict) -> str | None:
    try:
        return exposure_problem(rig)
    except OSError as e:
        # uvc-util failing says nothing about the frames, so the camera stays on unverified
        return f"exposure unverified: {e}"


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
        frames = CameraSource(open_camera(cfg["rig"]), process=Pipeline(cfg, lens=load_lens()).step)
    except (RuntimeError, OSError) as e:
        print(f"camera off: {e}")
        frames = None
    exposure = check_exposure(cfg["rig"]) if frames is not None else None
    actuator = None
    if cfg["rig"]["actuator"] == "real":
        actuator = SerialActuator(open_board(cfg["rig"]), cfg["rig"])
    loop = Loop(cfg, hand, frames, actuator, exposure)
    start_run_log(loop.bus)
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
