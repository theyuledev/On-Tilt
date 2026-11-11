"""Draws the 3D plate in node on a pen that writes down every call, in order.

A call is ["name", ...args], a property set is ["name=", value] with "gradient" for a gradient.
"""

import json
import subprocess

from src.web.server import STATIC

SCRIPTS = ("race", "gates", "hud", "cam", "follow", "lens", "plate3d")
PX, HALF, BALL_R = 900, 135, 8
CFG = {"plate_inner_mm": 2 * HALF, "ball_d_mm": 2 * BALL_R}
# each job is [js that gives a function, its args]; "PEN" in the args is the canvas stand-in and
# makes the answer what was drawn instead of what came back
RUN = """
const fs = require("fs"), vm = require("vm");
const box = vm.createContext({ devicePixelRatio: 1, performance: { now: () => 0 } });
for (const f of process.argv.slice(2)) vm.runInContext(fs.readFileSync(f, "utf8"), box);
const px = Number(process.argv[1]);
console.log(JSON.stringify(JSON.parse(fs.readFileSync(0, "utf8")).map(([js, args]) => {
  const drawn = [];
  const pen = new Proxy({ canvas: { clientWidth: px, width: px, height: px } }, {
    get: (t, k) => t[k] ?? ((...a) => {
      drawn.push([k, ...a]);
      return k === "measureText" ? { width: 10 * a[0].length } : { addColorStop() {} };
    }),
    set: (t, k, v) => !!drawn.push([k + "=", typeof v === "object" ? "gradient" : v]),
  });
  const out = vm.runInContext(js, box)(...args.map(a => (a === "PEN" ? pen : a)));
  return args.includes("PEN") ? drawn : out;
})));
"""


def run(*jobs):
    out = subprocess.run(
        ["node", "-e", RUN, str(PX), *(str(STATIC / f"{f}.js") for f in SCRIPTS)],
        input=json.dumps(jobs),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def snap(ball=(0, 0), valid=True, mode="hand", tilt=(0, 0)):
    return {
        "ball": {"x_mm": ball[0], "y_mm": ball[1], "valid": valid},
        "plate_state": {"sent_x_deg": tilt[0], "sent_y_deg": tilt[1]},
        "choices": {"mode": mode},
        "target": {"mm": [0, 0]},
        "race": None,
    }


def shots(at=(0, 0)):
    """The camera each shot settles on with the ball at `at` heading up the plate."""
    names = ("tower", "wide", "chase", "gate")
    cams = run(*[("shotGoal", [n, 0, list(at), [0, 1], None]) for n in names])
    return dict(zip(names, cams, strict=True))


def draw(maze, s, cam):
    return run(("draw3d", ["PEN", CFG, maze, s, cam]))[0]
