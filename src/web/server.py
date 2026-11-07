from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

STATIC = Path(__file__).parent / "static"


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

    @app.get("/camera.jpg")
    def camera():
        if loop.frame is None or loop.camera_lost:
            return Response(status_code=404)
        _, jpg = cv2.imencode(".jpg", loop.frame.image)
        return Response(jpg.tobytes(), media_type="image/jpeg")

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
