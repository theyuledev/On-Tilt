import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

STATIC = Path(__file__).parent / "static"
JPEG_SCALE = 0.5  # 1280x800 camera -> 640x400 on the page
JPEG_PERIOD_S = 1 / 15  # one encode per ~66 ms, the control loop shares this process


def make_app(loop) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/config")
    def config():
        rig, game = loop.cfg["rig"], loop.cfg["game"]
        return {
            "plate_inner_mm": rig["plate_inner_mm"],
            "ball_d_mm": rig["ball_d_mm"],
            "heartbeat_stale_s": game["heartbeat_stale_s"],
        }

    jpeg = {"t": float("-inf"), "bytes": b""}

    @app.get("/camera.jpg")
    def camera():
        if loop.frame is None or loop.camera_lost:
            return Response(status_code=404)
        now = time.monotonic()
        if now - jpeg["t"] >= JPEG_PERIOD_S:
            small = cv2.resize(loop.frame.image, None, fx=JPEG_SCALE, fy=JPEG_SCALE)
            _, jpg = cv2.imencode(".jpg", small)
            jpeg["t"], jpeg["bytes"] = now, jpg.tobytes()
        return Response(jpeg["bytes"], media_type="image/jpeg")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                msg = await sock.receive_json()
                if "x" in msg and "y" in msg:
                    loop.keys.set(float(msg["x"]), float(msg["y"]))
                if msg.get("arm"):
                    loop.arm()
                if msg.get("kill"):
                    loop.kill()
                await sock.send_json(loop.snapshot())
        except WebSocketDisconnect:
            pass

    return app
