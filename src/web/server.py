from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

STATIC = Path(__file__).parent / "static"


def make_app(loop) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

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
