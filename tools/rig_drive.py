"""Runs a balance test on the rig by itself: sets gains, arms, walks the target around the board
(far hops are the run-ups), holds at each, and writes down every trial. Someone stays in the
room with the kill.

    python tools/rig_drive.py <plan.json> <folder for trials.jsonl> [port]

plan.json: {"minutes": 12, "hold_s": 6, "reach_s": 8, "span_mm": 90,
            "configs": [{"name": "shipped", "set": {"control.kp_deg_mm": 0.05, ...}}, ...]}

It stops for good when: the time is up, a person kills or disarms, the watch rules trip
(tools/rig_watch.py), the app stops answering, or the ball is lost three times in one config.
A ball trapped in a dent (the plate stops on "ball is not answering the plate") is logged as stuck;
it asks for the ball to be rolled clear and carries on, up to six times a config.
A plan may list its own spots as "points_mm": [[x, y], ...] in place of the walk times span_mm.
A lost ball is announced; it re-arms only after the ball has sat still on the board for 3 s.
Settings it changed are put back at the end, in the app and in the local settings file.
"""

import asyncio
import json
import math
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import websockets

from src.control.pulse_map import center
from src.io.settings import LOCAL, load
from src.safety.watch import HZ, STOP, problems, row
from tools.rig_watch import say

# far diagonals, sides, short hops, centre: in units of span_mm
WALK = [
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (0.4, 0.3),
    (-0.3, 0.4),
    (0, 0),
]


STUCK = "ball is not answering the plate"


class Stop(Exception):
    pass


class Rig:
    def __init__(self, ws, cfg, port):
        self.ws, self.cfg, self.port = ws, cfg, port
        self.mid, self.rows, self.s = center(cfg["rig"]), [], None

    async def send(self, msg: dict) -> dict:
        await self.ws.send(json.dumps(msg))
        self.s = json.loads(await asyncio.wait_for(self.ws.recv(), 1.0))
        self.rows = [*self.rows, row(self.s, self.mid)][-8 * HZ :]
        return self.s

    async def tick(self, expect_armed: bool) -> dict:
        # no keys held, sent every tick: the app levels the plate when this heartbeat stops
        s = await self.send({"x": 0.0, "y": 0.0})
        found = problems(self.rows, self.cfg["rig"], self.cfg["control"])
        bad = [p for p in found if p in STOP or p == "board link lost"]
        if bad:
            await self.send({"kill": True})
            raise Stop(bad[0])
        if expect_armed and not s["armed"]:
            raise Stop(s.get("stopped_by") or "someone disarmed it")
        await asyncio.sleep(1 / HZ)
        return s

    async def settle_off(self) -> None:
        for _ in range(10 * HZ):
            s = await self.send({})
            if not s["armed"] and not s["killing"]:
                return
            await asyncio.sleep(1 / HZ)
        raise Stop("would not disarm")

    def post(self, changes: dict) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/settings.json",
            data=json.dumps(changes).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3).read()

    async def ball_resting(self, inner: float) -> None:
        """Waits, disarmed, for a ball that has sat still on the board for 3 s."""
        still = blind = 0
        while still < 3 * HZ:
            b = (await self.tick(False))["ball"]
            calm = b["valid"] and max(abs(b["x_mm"]), abs(b["y_mm"])) < inner
            still = still + 1 if calm and math.hypot(b["vx_mm_s"], b["vy_mm_s"]) < 15 else 0
            blind = 0 if calm else blind + 1
            if blind % (10 * HZ) == 5 * HZ:
                say("no ball on the board. put it in the middle")

    async def home(self) -> None:
        """Before a planned disarm: brings the ball to the middle and waits until it has slowed
        down there. A ball still rolling when the plate levels rolls straight off the board."""
        if not self.s or not self.s["armed"]:
            return
        await self.send({"target": [0.0, 0.0]})
        calm = 0
        for _ in range(12 * HZ):
            b = (await self.tick(True))["ball"]
            near = b["valid"] and math.hypot(b["x_mm"], b["y_mm"]) < 30
            calm = calm + 1 if near and math.hypot(b["vx_mm_s"], b["vy_mm_s"]) < 15 else 0
            if calm >= HZ:
                return

    async def moved_from(self, at: tuple[float, float]) -> None:
        """Waits, disarmed, until the ball is seen 20 mm from at: someone has rolled it clear."""
        n = 0
        while True:
            b = (await self.tick(False))["ball"]
            if b["valid"] and math.hypot(b["x_mm"] - at[0], b["y_mm"] - at[1]) > 20:
                return
            n += 1
            if n % (10 * HZ) == 0:
                say("ball stuck. roll it away from that spot")


async def one_config(rig: Rig, conf: dict, plan: dict, log, ends: float) -> None:
    inner = rig.cfg["rig"]["plate_inner_mm"] / 2 - rig.cfg["rig"]["ball_d_mm"]
    span = plan.get("span_mm", 0)
    todo = [list(p) for p in plan.get("points_mm") or [(ux * span, uy * span) for ux, uy in WALK]]
    lost = stuck = 0
    # arrived = inside 10 mm, or inside the wake line of a rest zone wider than that
    resting = conf["set"].get("control.balance_rest_mm", 0) > 0
    close = max(10.0, conf["set"].get("control.balance_rest_out_mm", 0.0) if resting else 0.0)
    await rig.send({"kill": True})
    await rig.settle_off()
    rig.post(conf["set"])
    await rig.send({"set": {"mode": "balance"}})
    while todo and time.monotonic() < ends:
        await rig.ball_resting(inner)
        say(f"arming, {conf['name']}")
        await asyncio.sleep(2.0)
        await rig.send({"arm": True})
        for _ in range(HZ):  # the snapshot shows it a tick or two later
            if (await rig.tick(False))["armed"]:
                break
        else:
            raise Stop(f"would not arm: {rig.s.get('arm_blocked')}")
        while todo and time.monotonic() < ends:
            goal = todo.pop(0)
            start = (await rig.send({"target": goal}))["t_s"]
            reached, blind, why = None, 0, None
            try:
                while rig.s["t_s"] - start < plan["reach_s"] + plan["hold_s"]:
                    s = await rig.tick(True)
                    b = s["ball"]
                    blind = 0 if b["valid"] else blind + 1
                    if blind > HZ:
                        why = "lost"
                        break
                    near = (
                        b["valid"] and math.hypot(b["x_mm"] - goal[0], b["y_mm"] - goal[1]) < close
                    )
                    reached = reached or (s["t_s"] if near else None)
                    if reached and s["t_s"] - reached >= plan["hold_s"]:
                        break
            except Stop as e:
                # a ball trapped in a dent stops the plate: that spot is data, not the end
                if str(e) != STUCK:
                    raise
                why = "stuck"
            log(
                {
                    "config": conf["name"],
                    "set": conf["set"],
                    "target": goal,
                    "t_start": start,
                    "t_end": rig.s["t_s"],
                    "reached_s": reached and reached - start,
                    "ball_lost": why == "lost",
                    "stuck": why == "stuck",
                }
            )
            if why == "lost":
                lost += 1
                await rig.send({"kill": True})
                await rig.settle_off()
                say("ball off the board. put it back")
                if lost >= 3:
                    raise Stop(f"ball lost three times on {conf['name']}")
                break
            if why == "stuck":
                stuck += 1
                at = rig.rows[-1]["pos"]
                await rig.settle_off()
                if stuck >= 6:
                    raise Stop(f"ball stuck six times on {conf['name']}")
                say("ball stuck. roll it away from that spot")
                await rig.moved_from(at)
                break
    await rig.home()
    await rig.send({"kill": True})


async def main() -> None:
    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]) / "trials.jsonl"
    port = sys.argv[3] if len(sys.argv) > 3 else "8000"
    cfg = load()
    backup = out.parent / "settings.local.before_drive.toml"
    shutil.copy(LOCAL, backup)
    keys = sorted({k for c in plan["configs"] for k in c["set"]})
    before = {k: cfg[k.split(".")[0]][k.split(".")[1]] for k in keys}
    ends = time.monotonic() + 60 * plan["minutes"]
    why = "time is up"
    with out.open("a") as f:

        def log(rec: dict) -> None:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(json.dumps(rec), flush=True)

        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            rig = Rig(ws, cfg, port)
            try:
                while time.monotonic() < ends:
                    for conf in plan["configs"]:
                        if time.monotonic() < ends:
                            await one_config(rig, conf, plan, log, ends)
            except Stop as e:
                why = str(e)
            except (TimeoutError, OSError, websockets.exceptions.WebSocketException) as e:
                why = f"the app stopped answering: {e!r}"
            finally:
                try:
                    await rig.send({"kill": True})
                    await rig.settle_off()
                    rig.post(before)
                except Exception as e:  # the file is still put back below
                    print(f"could not put the app's settings back: {e!r}")
                shutil.copy(backup, LOCAL)
                log({"ended": why, "settings_restored": before})
    print(f"test over: {why}")
    say(f"test over. {why}")


if __name__ == "__main__":
    asyncio.run(main())
