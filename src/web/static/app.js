const watch = new URLSearchParams(location.search).has("watch");
const held = new Set();
const keymap = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1],
                 KeyA: [-1, 0], KeyD: [1, 0], KeyW: [0, 1], KeyS: [0, -1] };
let pending = {};
let ws, cfg, maze, lastT, lastSeen, handWasSeen = false;
let armedNow = false, balancing = false, askedX = -Infinity;
const $ = id => document.getElementById(id);
const cam = $("cam"), nocam = $("nocam"), ctx = $("race").getContext("2d");
const padCtx = $("pad").getContext("2d");
const nextCam = () => { cam.src = "/camera.jpg?" + performance.now(); };
let camRetry;
const handcam = $("handcam"), handnone = $("handnone");
const nextHand = () => { handcam.src = "/hand.jpg?" + performance.now(); };
let handRetry;
const typing = e => e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA";
// choice groups on the page and the field each one sets
const PICKS = { modes: "mode", steers: "steer", handmodes: "hand_mode" };

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
      if (!$("racebtn").disabled) pending.race = raceCmd();
    }
  });
  addEventListener("keyup", e => { if (!typing(e) && clickKeys.has(e.code)) e.preventDefault(); held.delete(e.code); });
  addEventListener("blur", () => held.clear());
  $("arm").onclick = e => { pending.arm = true; e.currentTarget.blur(); };
  $("kill").onclick = e => { pending.kill = true; e.currentTarget.blur(); };
  $("racebtn").onclick = e => { pending.race = raceCmd(); e.currentTarget.blur(); };
  for (const [group, field] of Object.entries(PICKS)) {
    for (const b of $(group).children) b.onclick = e => { pick(field, b.dataset.pick); e.currentTarget.blur(); };
  }
  $("guard").onclick = e => {
    pick("guard", e.currentTarget.getAttribute("aria-checked") !== "true");
    e.currentTarget.blur();
  };
  for (const b of $("targets").children) b.onclick = e => { pending.target = b.dataset.pick; e.currentTarget.blur(); };
  $("race").onclick = e => {
    const c = e.currentTarget;
    if (balancing) pending.target = plateMm(e.offsetX / c.clientWidth, e.offsetY / c.clientHeight, cfg);
  };
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

function light(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className = "light " + cls;
}

function pick(field, value) {
  pending.set = { ...pending.set, [field]: value };
}

// the live choices on their buttons; they only move while disarmed, so they lock when armed
function choices(s) {
  const c = s.choices, locked = watch || !!s.armed;
  for (const [group, field] of Object.entries(PICKS)) {
    for (const b of $(group).children) {
      b.setAttribute("aria-pressed", b.dataset.pick === c[field]);
      b.disabled = locked || (field === "steer" && b.dataset.pick === "hand" && !cfg.hand);
    }
  }
  $("choices").classList.toggle("locked", locked);
  $("handrow").hidden = c.steer === "keys";
  // the guard only works on a player, and a build without one has nothing to switch
  $("guardsay").textContent = c.mode === "hand" ? "invisible walls" : `off in ${c.mode}`;
  $("guard").setAttribute("aria-checked", c.guard === true && c.mode === "hand");
  $("guard").disabled = locked || c.guard === null || c.mode !== "hand";
}

// balance swaps steering for the target picker; the target may move while armed
function targets(s) {
  const t = s.target;
  balancing = s.choices.mode === "balance";
  $("targetfield").hidden = !balancing;
  $("steerfield").hidden = balancing;
  $("race").style.cursor = balancing && !watch ? "crosshair" : "";
  $("sub").textContent = balancing ? "BALANCE · CLICK THE BOARD TO MOVE THE TARGET" : subtitle(maze);
  handnone.textContent = balancing ? "not used" : "no hand";
  for (const b of $("targets").children) {
    b.setAttribute("aria-pressed", b.dataset.pick === t.src);
    b.disabled = watch;
  }
  // keep looking for the X until the camera has one
  if (balancing && !watch && t.src === "marker" && !t.found && performance.now() - askedX > 1000) {
    askedX = performance.now();
    pending.target = "marker";
  }
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
  const gates = maze ? maze.checkpoints.length : 0;
  const g = guide(s, { live, watch, hand: cfg.hand, lost, gates });
  $("banner").textContent = g.text;
  $("banner").dataset.tone = g.tone;
}

function lights(s) {
  const b = s.ball;
  light("ballpill", b.valid ? "BALL" : "NO BALL", b.valid ? "on" : "warn");
  light("board", s.board_lost ? "BOARD LOST" : "BOARD", s.board_lost ? "warn" : "on");
  light("camlight", s.camera_lost ? "CAM LOST" : cam.hidden ? "NO CAM" : "CAM",
        s.camera_lost ? "warn" : cam.hidden ? "" : "on");
  light("hand", !s.hand_seen ? "NO HAND" : s.hand_calibrating ? "HOLD STILL" : "HAND",
        !s.hand_seen ? "" : s.hand_calibrating ? "warn" : "on");
  $("stale").hidden = $("level").hidden = !(s.stale && s.armed && !s.hand_calibrating);
  nocam.textContent = s.camera_lost ? "camera lost" : "no camera";
  $("armed").textContent = s.killing ? "KILLING" : s.armed ? "ARMED" : "DISARMED";
  $("armed").className = "chip " + (s.killing ? "warn" : s.armed ? "on" : "");
  // leaving the page would leave an armed plate with no kill key
  $("settings").classList.toggle("off", !!s.armed);
}

function show(s) {
  armedNow = !!s.armed;
  // HAND LOST only when a hand that was steering this arm drops out
  if (!s.armed) handWasSeen = false; else if (s.hand_seen) handWasSeen = true;
  banner(s, true, s.armed && handWasSeen && !s.hand_seen);
  if (!s.plate_state) return;
  if (s.t_s !== lastT) { lastT = s.t_s; lastSeen = performance.now(); }
  const b = s.ball, p = s.plate_state;
  lights(s);
  $("ballx").textContent = "X" + (b.valid ? signed(b.x_mm, 1) : "--.-");
  $("bally").textContent = "Y" + (b.valid ? signed(b.y_mm, 1) : "--.-");
  $("tiltx").textContent = `X ${signed(p.sent_x_deg)}°`;
  $("tilty").textContent = `Y ${signed(p.sent_y_deg)}°`;
  $("arm").disabled = watch || !!s.armed;
  $("racebtn").disabled = watch || s.race === undefined || s.choices.mode === "balance";
  choices(s);
  targets(s);
  ring(s.ball_px);
  strip(cfg, s);
  hud(s, maze);
  results(s);
  draw(ctx, cfg, maze, s);
  drawPad(padCtx, cfg, s);
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { light("link", "LINK", "on"); nextCam(); nextHand(); frame(); };
  ws.onclose = () => { light("link", "OFFLINE", "warn"); banner(null, false, false); setTimeout(connect, 1000); };
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
    light("beat", `${Math.round(age * 1000)} MS`, age > cfg.heartbeat_stale_s ? "warn" : "on");
  }
  requestAnimationFrame(beat);
}

function subtitle(maze) {
  if (!maze) return "NO MAZE ON THE BOARD";
  if (maze.walls && maze.walls.length) return `MAZE READ FROM BOARD · ${maze.walls.length} STROKES`;
  return `COURSE LOADED · ${maze.checkpoints.length} GATES`;
}

const json = url => fetch(url).then(r => r.ok ? r.json() : Promise.reject(new Error(`${url} ${r.status}`)));
// no maze just means no race, the plate still has to arm and kill
Promise.all([json("/config"), json("/race/maze").catch(() => null)]).then(([c, m]) => {
  cfg = c;
  maze = m;
  $("scale").textContent = `1 px = ${(1 / cfg.px_per_mm).toFixed(2)} mm · ${cfg.camera_fps} fps`;
  connect();
  beat();
}).catch(e => { $("banner").textContent = `load failed: ${e.message}`; });
