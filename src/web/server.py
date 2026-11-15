import asyncio
import math
import re
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src import switches
from src.contracts import Choices
from src.games import maze
from src.inputs import hand_view
from src.modes import spot
from src.vision.cross import cross_mm
from src.vision.line_path import LinePath, agree, find_line
from src.vision.px_to_mm import outline_px, with_calib
from src.web import form

STATIC = Path(__file__).parent / "static"
JPEG_SCALE = 0.5  # 1280x800 camera -> 640x400 on the page
JPEG_PERIOD_S = 1 / 15  # one encode per ~66 ms, the control loop shares this process
FULL_JPEG_QUALITY = 95  # calibration fits the lens from these, keep the pixels honest
NAME_MAX = 8  # same as the name field maxlength, the name ends up in a file name
LINE_READS = 5  # pictures the line is read off, the path most of them agree on is kept
ARMED, NO_PICTURE = "armed, kill first", "no camera picture"


def clean_name(raw) -> str:
    return re.sub(r"[^A-Za-z0-9 _-]", "", str(raw)).strip()[:NAME_MAX].strip()


def retarget(loop, want) -> None:
    # the X is looked for on the newest overhead frame; no camera means no X
    frame = getattr(loop, "frame", None)
    look = want == "marker" and frame is not None
    spot.aim(loop.cfg, want, cross_mm(frame.image, loop.cfg["vision"]) if look else None)


def read_line(loop, pipe) -> LinePath:
    """The drawn line off the next LINE_READS pictures, on the board map the tracker uses."""
    reads, last, rig = [], None, loop.cfg["rig"]
    while len(reads) < LINE_READS:
        frame = loop.frame
        if loop.safety.armed:
            return LinePath([], ARMED)
        old = frame is None or time.monotonic() - frame.t_s > rig["stale_intent_s"]
        if pipe is None or old or loop.camera_lost:
            return LinePath([], NO_PICTURE)
        if frame is last:
            time.sleep(1 / rig["camera_fps"])
            continue
        last, edge = frame, pipe.cfg["line_edge_mm"]
        reads.append(find_line(frame.image, pipe.cfg, pipe.half_mm, edge, loop.ball_px))
    return agree(reads, pipe.cfg)


def use_line(loop, got: LinePath) -> str:
    """A path read becomes the one path mode follows; anything else changes nothing."""
    if loop.safety.armed:
        return ARMED
    if got.reason:
        return got.reason
    loop.cfg["control"]["path_mm"] = got.points
    switches.choose(loop, Choices(mode="path"))
    return "ok"


def make_app(loop) -> FastAPI:
    app = FastAPI()
    app.state.restart, app.state.leave = False, lambda: None
    # the pipeline the camera feeds, for the board map it tracks; the last line read
    app.state.pipe = app.state.line = None
    vision = with_calib(loop.cfg["vision"])
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
            "plate_outline_px": outline_px(rig["plate_inner_mm"] / 2, vision),
            "heartbeat_stale_s": game["heartbeat_stale_s"],
            "dead_zone": inp["dead_zone"],
            "hand": loop.hand is not None,
            "camera": getattr(loop, "frames", None) is not None,
            "tilt_full_deg": inp["intent_gain_deg"],
            "px_per_mm": loop.cfg["vision"]["px_per_mm"],
            "camera_fps": rig["camera_fps"],
            "checkpoint_r_mm": game["checkpoint_r_mm"],
            "path_mm": loop.cfg["control"]["path_mm"],
        }

    @app.get("/settings")
    def settings_page():
        return FileResponse(STATIC / "settings.html")

    @app.get("/settings.json")
    def settings_view():
        return form.view(loop)

    @app.post("/settings.json")
    async def settings_save(changes: dict):
        # on the loop's own thread, so no tick runs between the armed check and the change
        if loop.safety.armed:
            return JSONResponse({"error": "armed, kill first"}, status_code=409)
        try:
            form.save(loop, changes)
        except (ValueError, OSError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return form.view(loop)

    @app.post("/restart")
    async def restart():
        # a dead camera only comes back in a fresh process: it has to open before the board does
        if loop.safety.armed:
            return JSONResponse({"error": "armed, kill first"}, status_code=409)
        app.state.restart = True
        app.state.leave()
        return {"restarting": True}

    @app.get("/race/maze")
    def race_maze():
        try:
            m = maze.load(loop.cfg["game"]["maze"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"no maze: {e!r}")
            return Response(status_code=404)
        return {
            "points": [list(p) for p in m.points],
            "checkpoints": list(m.checkpoints),
            "start": list(m.start),
            "walls": [[list(p) for p in w] for w in m.walls],
        }

    async def line_read():
        try:
            got = await asyncio.to_thread(read_line, loop, app.state.pipe)
            # back on the loop's own thread, so no tick runs between the armed check and the change
            app.state.line = use_line(loop, got)
        except Exception as e:
            print(f"line read failed: {e!r}")
            app.state.line = "could not read the line"

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
        outline = None
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
                    if "set" in msg:
                        switches.choose(loop, switches.parse(msg["set"]))
                    if "target" in msg:
                        # finding the X takes a while, the plate must not wait for it
                        await asyncio.to_thread(retarget, loop, msg["target"])
                    if msg.get("line") and app.state.line != "reading":
                        # a read takes seconds, the socket keeps answering while it runs
                        app.state.line = "reading"
                        app.state.reading = asyncio.create_task(line_read())
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
                snap = loop.snapshot()
                if app.state.line is not None:
                    snap["line"] = app.state.line
                # the board outline is long, so a socket only gets it again when it moved
                if snap.get("plate_outline_px") is outline:
                    snap.pop("plate_outline_px", None)
                else:
                    outline = snap["plate_outline_px"]
                await sock.send_json(snap)
        except WebSocketDisconnect:
            pass

    return app
