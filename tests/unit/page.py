"""Runs the page's own scripts in node against a small stand-in for the browser.

The stand-in builds its elements from index.html, so a script that asks for an id the page
does not have fails here. Canvas drawing goes nowhere, everything else is the real code.
"""

import copy
import json
import subprocess
from functools import cache

from fastapi.testclient import TestClient

from src.games import maze, scoring
from src.games.session import Game
from src.io.settings import load
from src.loop import Loop
from src.web.server import STATIC, make_app

RUN = r"""
const fs = require("fs"), vm = require("vm"), path = require("path");
const dir = process.argv[1], job = JSON.parse(fs.readFileSync(0, "utf8"));
const els = {}, win = {}, sockets = [], scripts = [];
let clock = 0, rafs = [];
const VOID = new Set(["meta", "link", "img", "input"]);
const pen = e => new Proxy({ canvas: e }, {
  get: (t, k) => (k in t ? t[k] : () => ({ width: 0 })),
  set: (t, k, v) => ((t[k] = v), true),
});

function el(tag) {
  const e = {
    tagName: tag.toUpperCase(), textContent: "", className: "", hidden: false, disabled: false,
    value: "", dataset: {}, style: {}, attrs: {}, children: [], listeners: {},
    clientWidth: 300, clientHeight: 48, offsetWidth: 0,
    getContext() { return (this.pen ||= pen(this)); },
    getAttribute(k) { return this.attrs[k] ?? null; },
    setAttribute(k, v) {
      this.attrs[k] = String(v);
      if (k === "id") els[v] = this;
      else if (k === "class") this.className = v;
      else if (k === "hidden" || k === "disabled") this[k] = true;
      else if (k === "value") this.value = v;
      else if (k === "src" && this.tagName === "SCRIPT") scripts.push(v);
      else if (k.startsWith("data-")) this.dataset[k.slice(5)] = v;
    },
    addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); },
    append(...kids) { this.children.push(...kids); },
    replaceChildren(...kids) { this.children = kids; },
    blur() {},
  };
  e.classList = {
    toggle(c, on) {
      const set = new Set(e.className.split(" ").filter(Boolean));
      if (on) set.add(c); else set.delete(c);
      e.className = [...set].join(" ");
    },
  };
  return e;
}

function parse(html) {
  const stack = [el("html")], attrs = String.raw`((?:\s+[\w:-]+(?:="[^"]*")?)*)`;
  const skip = String.raw`<!--[\s\S]*?-->|<!doctype[^>]*>`;
  const re = new RegExp(String.raw`${skip}|<\/(\w+)>|<(\w+)${attrs}\s*>|([^<]+)`, "gi");
  for (const m of html.matchAll(re)) {
    const top = stack[stack.length - 1];
    if (m[1]) stack.pop();
    else if (m[2]) {
      const e = el(m[2]);
      for (const a of m[3].matchAll(/([\w:-]+)(?:="([^"]*)")?/g)) e.setAttribute(a[1], a[2] ?? "");
      top.children.push(e);
      if (!VOID.has(m[2].toLowerCase())) stack.push(e);
    } else if (m[4] && m[4].trim()) top.textContent += m[4];
  }
  if (stack.length !== 1) throw new Error("index.html tags do not balance");
}

parse(fs.readFileSync(path.join(dir, "index.html"), "utf8"));
const box = vm.createContext({
  document: { getElementById: id => els[id] ?? null, createElement: el, body: el("body") },
  location: { search: job.query || "", host: "page" },
  URLSearchParams, console, devicePixelRatio: 1,
  performance: { now: () => clock },
  requestAnimationFrame: fn => { rafs.push(fn); },
  setTimeout: () => 0, clearTimeout: () => {},
  addEventListener: (type, fn) => { (win[type] ||= []).push(fn); },
  fetch: url => Promise.resolve({
    ok: job.routes[url] != null, status: 404, json: () => Promise.resolve(job.routes[url]),
  }),
  WebSocket: class {
    constructor() { this.readyState = 1; this.sent = []; sockets.push(this); }
    send(text) { this.sent.push(JSON.parse(text)); }
  },
});
for (const src of scripts) {
  const code = fs.readFileSync(path.join(dir, path.basename(src)), "utf8");
  vm.runInContext(code, box, { filename: src });
}

function key(type, code, on) {
  const e = { code, target: on ? els[on] : { tagName: "BODY" }, prevented: false,
              preventDefault() { this.prevented = true; } };
  const heard = on ? els[on].listeners[type] || [] : [];
  for (const fn of [...heard, ...(win[type] || [])]) fn(e);
  return e.prevented;
}

// "modes/1" is the second child of #modes
const find = at => { const [id, n] = at.split("/"); return n ? els[id].children[n] : els[id]; };
const acts = {
  snap: s => sockets[0].onmessage({ data: JSON.stringify(s) }),
  open: () => sockets[0].onopen(),
  close: () => sockets[0].onclose(),
  frame: () => { const run = rafs; rafs = []; run.forEach(fn => fn()); },
  wait: ms => { clock += ms; },
  down: ([code, on]) => key("keydown", code, on),
  up: ([code, on]) => key("keyup", code, on),
  click: at => { const e = find(at); if (!e.disabled) e.onclick({ currentTarget: e }); },
  type: ([id, text]) => { els[id].value = text; },
  // a click at u, v (0..1) across an element
  tap: ([id, u, v]) => {
    const e = els[id];
    e.onclick({ currentTarget: e, offsetX: u * e.clientWidth, offsetY: v * e.clientHeight });
  },
};
const view = e => ({
  text: e.textContent, cls: e.className, hidden: e.hidden, disabled: e.disabled, value: e.value,
  attrs: e.attrs, data: e.dataset, style: e.style, kids: e.children.map(view),
});

setImmediate(() => {
  const out = job.steps.map(([act, arg]) => acts[act](arg) ?? null);
  const page = Object.fromEntries(Object.entries(els).map(([id, e]) => [id, view(e)]));
  const sent = sockets.flatMap(s => s.sent), body = box.document.body.className;
  console.log(JSON.stringify({ out, page, sent, body }));
});
"""


@cache
def real() -> dict:
    """What the sim loop with the stock maze really serves: one snapshot, /config, /race/maze."""
    cfg = load()
    game = Game(cfg, maze.load(cfg["game"]["maze"]), scoring.step)
    loop = Loop(cfg, game=game)
    loop.keys.set(0.0, 0.0)  # the page sends its keys every frame
    game.tick(loop.tick(1.0, 0.01), False, False)
    client = TestClient(make_app(loop))
    return {
        "snap": loop.snapshot(),
        "/config": client.get("/config").json(),
        "/race/maze": client.get("/race/maze").json(),
    }


def snap(**over) -> dict:
    """The real snapshot (disarmed, race idle) with fields swapped; race=... drops the race."""
    s = {**copy.deepcopy(real()["snap"]), **over}
    return {k: v for k, v in s.items() if v is not ...}


def race(state, **over) -> dict:
    return {**real()["snap"]["race"], "state": state, **over}


def run(steps, *, query="", config=None, **routes) -> dict:
    """Loads the page, then plays steps: ["snap", {...}], ["down", ["Space", None]], ..."""
    served = {k: real()[k] for k in ("/config", "/race/maze")}
    served["/config"] = {**served["/config"], **(config or {})}
    served.update({f"/{k.replace('_', '/')}": v for k, v in routes.items()})
    job = {"query": query, "routes": served, "steps": [list(s) for s in steps]}
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC)],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)
