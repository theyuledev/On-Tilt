const watch = new URLSearchParams(location.search).has("watch");
const held = new Set();
const keymap = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1],
                 KeyA: [-1, 0], KeyD: [1, 0], KeyW: [0, 1], KeyS: [0, -1] };
let pending = {};
let ws, cfg, maze, lastT, lastSeen, handWasSeen = false;
const $ = id => document.getElementById(id);
const cam = $("cam"), nocam = $("nocam"), ctx = $("race").getContext("2d");
const padCtx = $("pad").getContext("2d");
const nextCam = () => { cam.src = "/camera.jpg?" + performance.now(); };
let camRetry;

if (watch) {
  // a projector tab must never overwrite the player's keys
  $("arm").disabled = $("kill").disabled = true;
} else {
  // a focused button clicks on enter or space keyup, so space after a kill could re-arm
  const clickKeys = new Set(["Space", "Enter", "NumpadEnter"]);
  addEventListener("keydown", e => {
    if (clickKeys.has(e.code)) e.preventDefault();
    if (e.code === "Space") { pending.kill = true; return; }
    if (e.code === "KeyC" && !e.metaKey && !e.ctrlKey) { pending.recentre = true; return; }
    if (keymap[e.code]) { held.add(e.code); e.preventDefault(); }
  });
  addEventListener("keyup", e => { if (clickKeys.has(e.code)) e.preventDefault(); held.delete(e.code); });
  addEventListener("blur", () => held.clear());
  $("arm").onclick = e => { pending.arm = true; e.currentTarget.blur(); };
  $("kill").onclick = e => { pending.kill = true; e.currentTarget.blur(); };
}
cam.onload = () => { cam.hidden = false; nocam.hidden = true; nextCam(); };
cam.onerror = () => {
  cam.hidden = true;
  nocam.hidden = false;
  clearTimeout(camRetry);
  camRetry = setTimeout(nextCam, 1000);
};

function intent() {
  let x = 0, y = 0;
  for (const k of held) { x += keymap[k][0]; y += keymap[k][1]; }
  return { x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)) };
}

function pill(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className = "pill " + cls;
}

function ring(px) {
  const el = $("seen");
  el.hidden = !px || cam.hidden;
  if (el.hidden) return;
  const k = cam.clientWidth / cfg.camera_width;
  el.style.left = `${k * px[0]}px`;
  el.style.top = `${k * px[1]}px`;
  el.style.width = el.style.height = `${2 * k * px[2]}px`;
}

function show(s) {
  if (!s.plate_state) return;
  if (s.t_s !== lastT) { lastT = s.t_s; lastSeen = performance.now(); }
  const b = s.ball, p = s.plate_state;
  pill("ballpill", "ball: " + (b.valid ? "tracked" : "lost"), b.valid ? "on" : "warn");
  $("speed").textContent = b.valid ? Math.round(Math.hypot(b.vx_mm_s, b.vy_mm_s)) : 0;
  $("timer").hidden = $("cps").hidden = s.race === undefined;
  // the banner only shouts when a hand that was steering this arm drops out
  if (!s.armed) handWasSeen = false; else if (s.hand_seen) handWasSeen = true;
  const lost = s.armed && handWasSeen && !s.hand_seen;
  $("banner").textContent = !s.armed ? "arm to race" : lost ? "HAND LOST" : "";
  $("banner").classList.toggle("lost", lost);
  pill("armed", s.killing ? "killing" : s.armed ? "armed" : "disarmed",
       s.killing ? "warn" : s.armed ? "on" : "");
  $("stale").hidden = $("level").hidden = !(s.stale && s.armed);
  $("board").hidden = !s.board_lost;
  pill("hand", "hand: " + (!s.hand_seen ? "none" : s.hand_calibrating ? "hold still" : "live"),
       !s.hand_seen ? "" : s.hand_calibrating ? "warn" : "on");
  nocam.textContent = s.camera_lost ? "camera lost" : "no camera";
  $("tilt").textContent = `tilt x ${p.sent_x_deg.toFixed(2)}°  y ${p.sent_y_deg.toFixed(2)}°`;
  ring(s.ball_px);
  strip(cfg, s);
  draw(ctx, cfg, maze, s);
  drawPad(padCtx, cfg, s);
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { pill("link", "live", "on"); nextCam(); frame(); };
  ws.onclose = () => { pill("link", "offline", "warn"); setTimeout(connect, 1000); };
  ws.onmessage = e => { show(JSON.parse(e.data)); requestAnimationFrame(frame); };
}

function frame() {
  if (ws.readyState !== 1) return;
  ws.send(JSON.stringify(watch ? {} : { ...intent(), ...pending }));
  pending = {};
}

function beat() {
  if (lastSeen !== undefined) {
    const age = (performance.now() - lastSeen) / 1000;
    pill("beat", `tick ${Math.round(age * 1000)} ms ago`, age > cfg.heartbeat_stale_s ? "warn" : "on");
  }
  requestAnimationFrame(beat);
}

const json = url => fetch(url).then(r => r.ok ? r.json() : Promise.reject(new Error(`${url} ${r.status}`)));
Promise.all([json("/config"), json("/race/maze")]).then(([c, m]) => {
  cfg = c;
  maze = m;
  $("cps").textContent = `0/${maze.checkpoints.length}`;
  connect();
  beat();
}).catch(e => { $("banner").textContent = `load failed: ${e.message}`; });
