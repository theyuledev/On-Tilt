// race readouts from s.race; app.js calls hud(s, maze) each snapshot (uses its $ and race.js fmtLap)
const AHEAD = "#3EE6E0";
const BEHIND = "#E69F00";
const PRE = ["calibrating", "staging"];
const LIVE = [...PRE, "countdown", "running"];
let prev = null;

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

const signed = (d, n = 2) => `${d < 0 ? "-" : "+"}${Math.abs(d).toFixed(n)}`;

function split(r, was) {
  const el = $("split"), first = typeof r.ghost_lap_s !== "number";
  if (r.reached > was.reached && (first || r.delta_s !== null)) {
    const d = r.delta_s;
    pop(el, first ? `CP ${r.reached}  ${r.elapsed_s.toFixed(2)}` : signed(d), "split");
    el.style.color = first || d === 0 ? "" : d < 0 ? AHEAD : BEHIND;
  } else if (r.state === "idle" || r.state === "countdown") {
    el.textContent = "";
  }
}

function penalty(r, was) {
  if (r.penalty_s > was.penalty_s) pop($("pen"), `+${(r.penalty_s - was.penalty_s).toFixed(1)} s`, "pen");
}

function result(r, was) {
  const el = $("result"), [head, sub] = el.children, edge = r.state !== was.state;
  el.dataset.state = r.state;
  if (r.state === "finished") {
    if (!edge) return;
    head.textContent = "FINISH";
    sub.textContent = fmtLap(r.score ? r.score.time_s : r.elapsed_s);
    // fades so the board shows again behind the name entry
    restart(el, "brief");
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
  el.hidden = !head.textContent;
}

// label, value and tone of the ghost row for the latest split
function versus(r) {
  const d = r ? r.delta_s : null, done = !!r && r.state === "finished";
  if (r && LIVE.includes(r.state) && typeof r.ghost_lap_s !== "number") return ["FIRST LAP", "--", ""];
  if (typeof d !== "number") return ["VS GHOST", "--", ""];
  if (d < 0) return [done ? "BEAT THE GHOST BY" : "AHEAD OF GHOST", signed(d), "ahead"];
  if (d > 0) return [done ? "BEHIND THE GHOST BY" : "BEHIND GHOST", signed(d), "behind"];
  return ["LEVEL WITH GHOST", signed(d), ""];
}

function lampClass(i, reached) {
  return i < reached ? "lamp hit" : i === reached ? "lamp next" : "lamp";
}

// one lamp per checkpoint, the last one is the finish
function lamps(n, reached) {
  const el = $("cps");
  if (el.children.length !== n) {
    el.replaceChildren(...Array.from({ length: n }, (_, i) => {
      const lamp = document.createElement("span");
      lamp.textContent = i === n - 1 ? "F" : String(i + 1);
      return lamp;
    }));
  }
  [...el.children].forEach((lamp, i) => { lamp.className = lampClass(i, reached); });
}

function readouts(s, r, maze) {
  const balance = s.choices.mode === "balance", free = !balance && s.race === undefined;
  const done = !!r && r.state === "finished";
  const counting = !!r && !PRE.includes(r.state) && r.state !== "idle";
  const t = done && r.score ? r.score.time_s : counting ? Math.max(0, r.elapsed_s ?? 0) : 0;
  $("scores").hidden = $("gates").hidden = balance || free;
  $("nomaze").hidden = !free;
  $("balance").hidden = !balance;
  if (balance) return balanceReadouts(s);
  $("timelabel").textContent = free ? "NOW PLAYING" : done ? "FINISH TIME" : "TIME";
  $("timer").className = free ? "free" : counting ? "" : "idle";
  $("timer").textContent = free ? "FREE PLAY" : fmtLap(t);
  $("best").textContent = r && typeof r.ghost_lap_s === "number" ? fmtLap(r.ghost_lap_s) : "--:--.--";
  const [say, value, tone] = versus(r);
  $("ghostsay").textContent = say;
  $("delta").textContent = value;
  $("ghostrow").className = `score ${tone}`;
  if (maze) lamps(maze.checkpoints.length, counting ? r.reached : 0);
}

function hud(s, maze) {
  const r = s.race && s.race.state ? s.race : null, live = !!r && LIVE.includes(r.state);
  document.body.classList.toggle("offpath", !!r && r.off_path === true);
  $("racebtn").textContent = live ? "ABORT" : "RACE";
  $("racebtn").classList.toggle("abort", live);
  readouts(s, r, maze);
  // no race text before the countdown
  if (!r || PRE.includes(r.state)) {
    prev = r;
    $("count").textContent = $("split").textContent = $("pen").textContent = "";
    for (const el of $("result").children) el.textContent = "";
    $("result").hidden = true;
    return;
  }
  const was = prev || r;
  countdown(r, was);
  split(r, was);
  penalty(r, was);
  result(r, was);
  prev = r;
}
