"""A second pair of eyes on a running rig session: reads the app's snapshots about 20 times a
second, says out loud when the ball is gone, and with --kill presses kill when the plate misbehaves.

    python tools/rig_watch.py            watch and speak only
    python tools/rig_watch.py --kill     also kill on a stop problem
    python tools/rig_watch.py --replay data/runs/<log>.jsonl     what it would have said, no rig

The app runs the same rules itself (src/safety/watch.py); this is a second look from outside in
case the app hangs. It only ever sends an empty message or a kill, never an arm.
"""

import asyncio
import json
import os
import subprocess
import sys
import time

from src.control.pulse_map import center
from src.io.settings import load
from src.safety.watch import HZ, STOP, problems, row


def say(text: str) -> None:
    # RIG_QUIET=1 for dry runs against the sim
    if not os.environ.get("RIG_QUIET"):
        subprocess.Popen(["say", text])


def replay(path: str, cfg: dict) -> None:
    mid = center(cfg["rig"])
    rows, seen, last_t, counts = [], {}, -1.0, {}
    for line in open(path):
        s = json.loads(line)
        if s["t_s"] - last_t < 1 / HZ:
            continue
        last_t = s["t_s"]
        rows = [*rows, row(s, mid)][-8 * HZ :]
        for p in problems(rows, cfg["rig"], cfg["control"]):
            if s["t_s"] - seen.get(p, -99) > 3:
                counts[p] = counts.get(p, 0) + 1
                at, (x, y) = rows[-1], rows[-1]["pos"]
                where = f"ball ({x:.0f},{y:.0f}) pulses {at['us']}"
                print(f"  t_s {s['t_s']:.1f} {s['mode']:8s} {p}  {where}")
            seen[p] = s["t_s"]
    print(f"{path.split('/')[-1]}: {counts or 'nothing to report'}")


async def watch(cfg: dict, kill: bool, port: str) -> None:
    import websockets

    mid = center(cfg["rig"])
    rows, spoken = [], {}
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
        while True:
            await ws.send("{}")
            s = json.loads(await asyncio.wait_for(ws.recv(), 1.0))
            if "t_s" in s:
                rows = [*rows, row(s, mid)][-8 * HZ :]
                for p in problems(rows, cfg["rig"], cfg["control"]):
                    if kill and p in STOP and rows[-1]["armed"]:
                        await ws.send(json.dumps({"kill": True}))
                        await ws.recv()
                    if time.monotonic() - spoken.get(p, -99) > 4:
                        spoken[p] = time.monotonic()
                        stop = kill and p in STOP
                        print(
                            f"{time.strftime('%H:%M:%S')} {p}{' -> KILL' if stop else ''}",
                            flush=True,
                        )
                        say(("stopped. " if stop else "") + p)
            await asyncio.sleep(1 / HZ)


def main() -> None:
    cfg = load()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--replay" in sys.argv:
        for path in args:
            replay(path, cfg)
        return
    try:
        asyncio.run(watch(cfg, "--kill" in sys.argv, args[0] if args else "8000"))
    except Exception as e:  # the watcher must never go quiet without saying so
        print(f"watcher stopped: {e!r}")
        say("the watcher lost the app")


if __name__ == "__main__":
    main()
