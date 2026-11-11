const watch = new URLSearchParams(location.search).has("watch");
const held = new Set();
const keymap = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1],
                 KeyA: [-1, 0], KeyD: [1, 0], KeyW: [0, 1], KeyS: [0, -1] };
let pending = {};
let ws, cfg, maze, lastT, lastSeen, handWasSeen = false;
let armedNow = false;
const $ = id => document.getElementById(id);
const cam = $("cam"), nocam = $("nocam"), ctx = $("race").getContext("2d");
const padCtx = $("pad").getContext("2d");
const nextCam = () => { cam.src = "/camera.jpg?" + performance.now(); };
let camRetry;
const handcam = $("handcam"), handnone = $("handnone");
const nextHand = () => { handcam.src = "/hand.jpg?" + performance.now(); };
let handRetry;
const typing = e => e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA";

if (watch) {
  // a projector tab must never overwrite the player's keys
  $("arm").disabled = $("kill").disabled = $("racebtn").disabled = true;
} else {
  // a focused button clicks on enter or space keyup, so space after a kill could re-arm
  const clickKeys = new Set(["Space", "Enter", "NumpadEnter"]);
  addEventListener("keydown", e => {
    if (e.code === "Space" && armedNow) { pending.kill = true; e.preventDefault(); return; }
    if (typing(e)) return;
    if (clickKeys.has(e.code)) e.preventDefault();
    if (e.code === "Space") { pending.kill = true; return; }
    if (e.code === "KeyC" && !e.metaKey && !e.ctrlKey) { pending.recentre = true; return; }
    if (keymap[e.code]) { held.add(e.code); e.preventDefault(); }
    if (e.code === "KeyR" && !(e.repeat || e.metaKey || e.ctrlKey || e.altKey || typing(e))) {
      pending.race = raceCmd();
    }
  });
  addEventListener("keyup", e => { if (!typing(e) && clickKeys.has(e.code)) e.preventDefault(); held.delete(e.code); });
  addEventListener("blur", () => held.clear());
  $("arm").onclick = e => { pending.arm = true; e.currentTarget.blur(); };
  $("kill").onclick = e => { pending.kill = true; e.currentTarget.blur(); };
  $("racebtn").onclick = e => { pending.race = raceCmd(); e.currentTarget.blur(); };
}
cam.onload = () => { cam.hidden = false; nocam.hidden = true; nextCam(); };
cam.onerror = () => {
  cam.hidden = true;
  nocam.hidden = false;
  clearTimeout(camRetry);
  camRetry = setTimeout(nextCam, 1000);
};
handcam.onload = () => { handcam.hidden = false; handnone.hidden = true; nextHand(); };
handcam.onerror = () => {
  handcam.hidden = true;
  handnone.hidden = false;
  clearTimeout(handRetry);
  handRetry = setTimeout(nextHand, 1000);
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

function banner(s, live, lost) {
  const g = guide(s, { live, watch, hand: cfg.hand, lost });
  $("banner").textContent = g.text;
  $("banner").dataset.tone = g.tone;
}

function show(s) {
  armedNow = !!s.armed;
  // HAND LOST only when a hand that was steering this arm drops out
  if (!s.armed) handWasSeen = false; else if (s.hand_seen) handWasSeen = true;
  banner(s, true, s.armed && handWasSeen && !s.hand_seen);
  if (!s.plate_state) return;
  if (s.t_s !== lastT) { lastT = s.t_s; lastSeen = performance.now(); }
  const b = s.ball, p = s.plate_state;
  pill("ballpill", "ball: " + (b.valid ? "tracked" : "lost"), b.valid ? "on" : "warn");
  $("speed").textContent = b.valid ? Math.round(Math.hypot(b.vx_mm_s, b.vy_mm_s)) : 0;
  $("timer").hidden = $("cps").hidden = s.race === undefined;
  pill("armed", s.killing ? "killing" : s.armed ? "armed" : "disarmed",
       s.killing ? "warn" : s.armed ? "on" : "");
  $("armblock").hidden = !s.arm_blocked;
  $("armblock").textContent = s.arm_blocked || "";
  $("stale").hidden = $("level").hidden = !(s.stale && s.armed && !s.hand_calibrating);
  $("board").hidden = !s.board_lost;
  pill("hand", "hand: " + (!s.hand_seen ? "none" : s.hand_calibrating ? "hold still" : "live"),
       !s.hand_seen ? "" : s.hand_calibrating ? "warn" : "on");
  nocam.textContent = s.camera_lost ? "camera lost" : "no camera";
  $("tilt").textContent = `tilt x ${p.sent_x_deg.toFixed(2)}°  y ${p.sent_y_deg.toFixed(2)}°`;
  $("handtilt").textContent = `plate x ${p.sent_x_deg.toFixed(2)}° y ${p.sent_y_deg.toFixed(2)}°`;
  ring(s.ball_px);
  strip(cfg, s);
  hud(s, maze);
  results(s);
  draw(ctx, cfg, maze, s);
  drawPad(padCtx, cfg, s);
  board(s, cfg);
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    pill("link", "live", "on");
    $("arm").disabled = watch;
    nextCam();
    nextHand();
    frame();
  };
  ws.onclose = () => {
    // one-shots clicked while offline must not fire on whatever app answers next, kill can wait
    delete pending.arm; delete pending.race; delete pending.recentre;
    pill("link", "offline", "warn"); banner(null, false, false);
    $("arm").disabled = true;
    setTimeout(connect, 1000);
  };
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
// no maze just means no race, the plate still has to arm and kill
Promise.all([json("/config"), json("/race/maze").catch(() => null)]).then(([c, m]) => {
  cfg = c;
  maze = m;
  if (maze) $("cps").textContent = `0/${maze.checkpoints.length}`;
  connect();
  beat();
}).catch(e => { $("banner").textContent = `load failed: ${e.message}`; });
