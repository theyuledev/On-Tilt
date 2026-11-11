import json
import shutil
import subprocess

import pytest

from src.web.server import STATIC

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

# loads the real page script with the browser stubbed out, then feeds the socket two bad states
RUN = """
const fs = require("fs"), vm = require("vm");
const stub = { get: () => any, apply: () => any, construct: () => any, set: () => true };
const any = new Proxy(function () {}, stub);
const on = {}, sent = [], errors = [], raf = [];
let sock;
class WebSocket {
  constructor() { sock = this; this.readyState = 1; }
  send(m) { sent.push(JSON.parse(m)); }
}
const box = {
  document: any, performance: any, location: { search: "", host: "x" }, URLSearchParams, WebSocket,
  addEventListener: (type, fn) => { on[type] = fn; }, fetch: () => new Promise(() => {}),
  requestAnimationFrame: fn => raf.push(fn), console: { error: e => errors.push(String(e)) },
};
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), box);
box.connect();
for (const data of ["not json", "null"]) try { sock.onmessage({ data }); } catch {}
on.keydown({ code: "Space", target: {}, preventDefault() {} });
raf.forEach(f => f());
console.log(JSON.stringify({ sent, errors }));
"""


def test_bad_states_still_send_kill_and_log_once():
    out = subprocess.run(
        ["node", "-e", RUN, str(STATIC / "app.js")],
        capture_output=True,
        text=True,
        check=True,
    )
    got = json.loads(out.stdout)
    assert got["sent"] == [{"x": 0, "y": 0, "kill": True}, {"x": 0, "y": 0}]
    assert len(got["errors"]) == 1
