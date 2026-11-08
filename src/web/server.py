import math
import re
import time
import tomllib
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.inputs import hand_view
from src.io.settings import ROOT

STATIC = Path(__file__).parent / "static"
MAZES = ROOT / "configs" / "mazes"
JPEG_SCALE = 0.5  # 1280x800 camera -> 640x400 on the page
JPEG_PERIOD_S = 1 / 15  # one encode per ~66 ms, the control loop shares this process
FULL_JPEG_QUALITY = 95  # calibration fits the lens from these, keep the pixels honest
NAME_MAX = 8  # same as the name field maxlength, the name ends up in a file name


def clean_name(raw) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]", "", str(raw)).strip()[:NAME_MAX].strip()


def make_app(loop) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/config")
    def config():
        rig, game, inp = loop.cfg["rig"], loop.cfg["game"], loop.cfg["input"]
        return {
            "plate_inner_mm": rig["plate_inner_mm"],
            "ball_d_mm": rig["ball_d_mm"],
            "camera_width": rig["camera_width"],
            "heartbeat_stale_s": game["heartbeat_stale_s"],
            "dead_zone": inp["dead_zone"],
            "hand": loop.hand is not None,
        }

    @app.get("/race/maze")
    def maze():
        with (MAZES / f"{loop.cfg['game']['maze']}.toml").open("rb") as f:
            m = tomllib.load(f)
        return {"points": m["points"], "checkpoints": m["checkpoints"], "start": m["start"]}

    jpeg = {"t": float("-inf"), "bytes": b""}

    @app.get("/camera.jpg")
    def camera(full: bool = False):
        if loop.frame is None or loop.camera_lost:
            return Response(status_code=404)
        if full:
            quality = [cv2.IMWRITE_JPEG_QUALITY, FULL_JPEG_QUALITY]
            _, jpg = cv2.imencode(".jpg", loop.frame.image, quality)
            return Response(jpg.tobytes(), media_type="image/jpeg")
        now = time.monotonic()
        if now - jpeg["t"] >= JPEG_PERIOD_S:
            small = cv2.resize(loop.frame.image, None, fx=JPEG_SCALE, fy=JPEG_SCALE)
            _, jpg = cv2.imencode(".jpg", small)
            jpeg["t"], jpeg["bytes"] = now, jpg.tobytes()
        return Response(jpeg["bytes"], media_type="image/jpeg")

    @app.get("/hand.jpg")
    def hand():
        # the hand process writes this file, an old one means it stopped
        try:
            path = hand_view.HAND_JPG
            if time.time() - path.stat().st_mtime > loop.cfg["game"]["heartbeat_stale_s"]:
                return Response(status_code=404)
            return Response(path.read_bytes(), media_type="image/jpeg")
        except FileNotFoundError:
            return Response(status_code=404)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                msg = await sock.receive_json()
                try:
                    if msg.get("kill"):
                        loop.kill()
                    if "x" in msg and "y" in msg:
                        x, y = float(msg["x"]), float(msg["y"])
                        if math.isfinite(x + y):
                            loop.keys.set(x, y)
                    if msg.get("arm") and not msg.get("kill"):
                        loop.arm()
                    game = getattr(loop, "game", None)
                    if game is not None and msg.get("race") == "start":
                        game.start()
                    elif game is not None and msg.get("race") == "abort":
                        game.abort()
                    if "save" in msg and game is not None and (name := clean_name(msg["save"])):
                        game.save(name)
                    if msg.get("recentre") and loop.hand is not None:
                        loop.hand.recentre()
                except Exception as e:
                    print(f"ws message dropped: {e!r}")
                await sock.send_json(loop.snapshot())
        except WebSocketDisconnect:
            pass

    return app
