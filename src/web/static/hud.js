// race overlay from s.race; app.js calls hud(s, maze) each snapshot (uses its $ and race.js fmtLap)
const AHEAD = "#56B4E9";
const BEHIND = "#E69F00";
const SPLIT_MS = 2500;  // length of the .split animation in style.css
const LIVE = ["calibrating", "staging", "countdown", "running"];
let prev = null, splitAt = -Infinity;

function restart(el, cls) {
  el.className = "hud";
  void el.offsetWidth;  // reflow so the css animation runs again
  el.className = `hud ${cls}`;
}

function pop(el, text, cls) {
  el.textContent = text;
  restart(el, cls);
}

function raceCmd() {
  if (!prev) return undefined;
  return LIVE.includes(prev.state) ? "abort" : "start";
}

function countdown(r, was) {
  const el = $("count");
  if (r.state === "countdown") {
    const n = String(Math.ceil(-r.elapsed_s));
    if (el.textContent !== n) pop(el, n, "tick");
  } else if (was.state === "countdown" && r.state === "running") {
    pop(el, "GO", "go");
  } else if (el.textContent !== "GO") {
    el.textContent = "";
  }
}

const signed = d => `${d < 0 ? "-" : "+"}${Math.abs(d).toFixed(2)}`;

function split(r, was, now) {
  const el = $("delta"), first = typeof r.ghost_lap_s !== "number";
  if (r.reached > was.reached && (first || r.delta_s !== null)) {
    const d = r.delta_s;
    pop(el, first ? `CP ${r.reached}  ${r.elapsed_s.toFixed(2)}` : signed(d), "split");
    el.style.color = first || d === 0 ? "" : d < 0 ? AHEAD : BEHIND;
    splitAt = now;
  } else if (r.state === "idle" || (r.state === "countdown" && !first)) {
    el.textContent = "";
  } else if (first && LIVE.includes(r.state) && now - splitAt > SPLIT_MS) {
    el.textContent = "FIRST LAP";
    el.className = "hud";
    el.style.color = "";
  }
}

function penalty(r, was) {
  if (r.penalty_s > was.penalty_s) pop($("pen"), `+${(r.penalty_s - was.penalty_s).toFixed(1)} s`, "pen");
}

function result(r, was) {
  const el = $("result"), [head, sub] = el.children, edge = r.state !== was.state;
  el.dataset.state = r.state;
  if (r.state === "finished") {
    head.textContent = "FINISH";
    sub.textContent = fmtLap(r.score ? r.score.time_s : r.elapsed_s);
    if (edge) restart(el, "end");
  } else if (r.state === "dnf") {
    head.textContent = "DNF";
    sub.textContent = `${r.reason}\narm, then R`;
    if (edge) restart(el, "end");
  } else if (edge && r.state === "idle" && r.reason) {
    head.textContent = r.reason === "left the start" ? "false start" : r.reason;
    sub.textContent = "";
    restart(el, "brief");
  } else if (r.state !== "idle") {
    head.textContent = sub.textContent = "";
  }
}

function hud(s, maze) {
  const r = s.race && s.race.state ? s.race : null, live = !!r && LIVE.includes(r.state);
  document.body.classList.toggle("racing", live);
  document.body.classList.toggle("offpath", !!r && r.off_path === true);
  $("racebtn").textContent = live ? "abort (r)" : "race (r)";
  if (!r) {
    prev = null;
    $("count").textContent = $("delta").textContent = $("pen").textContent = "";
    for (const el of $("result").children) el.textContent = "";
    return;
  }
  const was = prev || r;
  $("timer").textContent = fmtLap(Math.max(0, r.elapsed_s ?? 0));
  $("cps").textContent = `CP ${r.reached}/${maze.checkpoints.length}`;
  countdown(r, was);
  split(r, was, performance.now());
  penalty(r, was);
  result(r, was);
  prev = r;
}
