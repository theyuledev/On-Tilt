"""Draws the 3D plate in node on a pen that writes down every call, in order.

A call is ["name", ...args], a property set is ["name=", value] with "gradient" for a gradient.
"""

import json
import math
import subprocess

from src.web.server import STATIC

SCRIPTS = ("race", "board", "target", "gates", "hud", "cam", "follow", "lens", "plate3d", "deck3d")
PX, HALF, BALL_R, TAKE_R = 900, 135, 8, 30
OWN_PATH = [[50, 50], [-50, 50], [-50, -50]]
CFG = {
    "plate_inner_mm": 2 * HALF,
    "ball_d_mm": 2 * BALL_R,
    "checkpoint_r_mm": TAKE_R,
    "path_mm": OWN_PATH,
}
# each job is [js that gives a function, its args]; "PEN" in the args is the canvas stand-in and
# makes the answer what was drawn instead of what came back
RUN = """
const fs = require("fs"), vm = require("vm");
const Path2D = class { constructor(d) { this.d = d; } };
const box = vm.createContext({ devicePixelRatio: 1, performance: { now: () => 0 }, Path2D });
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


def run(*jobs, px=PX):
    out = subprocess.run(
        ["node", "-e", RUN, str(px), *(str(STATIC / f"{f}.js") for f in SCRIPTS)],
        input=json.dumps(jobs),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def snap(ball=(0, 0), valid=True, mode="hand", tilt=(0, 0), race=None):
    return {
        "ball": {"x_mm": ball[0], "y_mm": ball[1], "valid": valid},
        "plate_state": {"sent_x_deg": tilt[0], "sent_y_deg": tilt[1]},
        "choices": {"mode": mode},
        "target": {"mm": [0, 0]},
        "race": race,
        "t_s": 1.0,
    }


def shots(at=(0, 0)):
    """The camera each shot settles on with the ball at `at` heading up the plate."""
    names = ("tower", "wide", "chase", "gate")
    cams = run(*[("shotGoal", [n, 0, list(at), [0, 1], None]) for n in names])
    return dict(zip(names, cams, strict=True))


def draw(maze, s, cam, px=PX):
    return run(("draw3d", ["PEN", CFG, maze, s, cam]), px=px)[0]


def frames(maze, cam, *ss, view="draw3d"):
    """Several frames one after the other on the same page: what the last one drew."""
    look = cam if view == "draw3d" else {"x": 0, "y": 0, "zoom": 1, "tall": False}
    job = f"(pen, cfg, maze, look, ...ss) => ss.forEach(s => {view}(pen, cfg, maze, s, look))"
    drawn = run((job, ["PEN", CFG, maze, look, *ss]))[0]
    return drawn[max(i for i, call in enumerate(drawn) if call[0] == "clearRect") :]


def lines(drawn):
    return [call[1:3] for call in drawn if call[0] in ("moveTo", "lineTo")]


def joined(drawn, p, q):
    """Is a line drawn from p to q."""
    pts = lines(drawn)
    return any(
        math.dist(a, p) < 0.01 and math.dist(b, q) < 0.01
        for a, b in zip(pts, pts[1:], strict=False)
    )
