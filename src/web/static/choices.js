// the play choices on the page: mode, steering, hand mode, guard and the balance target
// (uses app.js $, pending, cfg, maze, watch, subtitle, handnone and target.js plateMm)
const PICKS = { modes: "mode", steers: "steer", handmodes: "hand_mode" };
let balancing = false, askedX = -Infinity, askedLine = -Infinity, wasReading = false;

function wireChoices() {
  for (const [group, field] of Object.entries(PICKS)) {
    for (const b of $(group).children) b.onclick = e => { pick(field, b.dataset.pick); e.currentTarget.blur(); };
  }
  $("guard").onclick = e => {
    pick("guard", e.currentTarget.getAttribute("aria-checked") !== "true");
    e.currentTarget.blur();
  };
  for (const b of $("targets").children) b.onclick = e => { pending.target = b.dataset.pick; e.currentTarget.blur(); };
  $("readline").onclick = e => {
    pending.line = true;
    // the click shows at once, before the app answers
    askedLine = performance.now();
    e.currentTarget.textContent = "READING...";
    e.currentTarget.disabled = true;
    e.currentTarget.blur();
  };
  $("race").onclick = e => {
    const c = e.currentTarget;
    // only the flat picture maps a click straight onto the plate
    if (!balancing || view !== "top") return;
    pending.target = plateMm(e.offsetX / c.clientWidth, e.offsetY / c.clientHeight, cfg);
  };
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
  readLine(s, locked);
}

// the line drawn on the board becomes the path; s.line is the last read's answer
function readLine(s, locked) {
  const reading = s.line === "reading" || performance.now() - askedLine < 700, say = $("linesay");
  const ok = s.line === "ok", pts = (s.path && s.path.points) || [];
  const cm = Math.round(pts.slice(1).reduce((a, p, i) => a + Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]), 0) / 10);
  $("readline").textContent = reading ? "READING..." : "READ LINE";
  $("readline").disabled = locked || reading || !cfg.camera || !!s.camera_lost;
  // its ends near the board edge can be off, so the ribbon is worth a look before arming
  say.textContent = ok ? `GOT IT: ${cm} cm of line. check the path on the board, then ARM`
    : s.line ? `NO PATH: ${s.line}` : "";
  say.classList.toggle("got", ok);
  say.classList.toggle("red", !ok && !!s.line);
  // a new answer flashes, so a second read that says the same is still seen
  if (wasReading && !reading && s.line) {
    say.classList.toggle("flash", false);
    void say.offsetWidth;
    say.classList.toggle("flash", true);
  }
  wasReading = reading;
  // the hint is for the path just read, not for a run or another mode
  say.hidden = !s.line || reading || (s.line === "ok" && (locked || s.choices.mode !== "path"));
}

// balance swaps steering for the target picker; the target may move while armed
function targets(s) {
  const t = s.target;
  balancing = s.choices.mode === "balance";
  $("targetfield").hidden = !balancing;
  $("steerfield").hidden = balancing;
  $("race").style.cursor = balancing && !watch && view === "top" ? "crosshair" : "";
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
