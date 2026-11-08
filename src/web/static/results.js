// finish panel in #side; app.js calls results(s) each snapshot
// (uses race.js fmtLap, hud.js signed/AHEAD/BEHIND and app.js pending)
const RES_DELAY_MS = 1000;  // the hud's big FINISH time shows first
const resEl = id => document.getElementById(id);
const resName = resEl("resname");
let resAt = null, resSaves = 0, resSaved = false;

resName.addEventListener("keydown", e => {
  if (e.code !== "Enter" && e.code !== "NumpadEnter") return;
  const name = resName.value.trim();
  if (!name) return;
  pending.save = name;
  resSaves += 1;
  resSaved = true;
  resName.blur();
  resEl("ressaved").hidden = false;
});

function resShow(r) {
  const first = typeof r.ghost_lap_s !== "number", d = r.delta_s, delta = resEl("resdelta");
  resAt = performance.now();
  resSaved = false;
  resEl("restime").textContent = fmtLap(r.score.time_s);
  resEl("respen").textContent = `penalty +${r.score.penalty_s.toFixed(1)} s`;
  delta.textContent = first || d === null ? "" : `${signed(d)} s`;
  delta.style.color = d === 0 ? "" : d < 0 ? AHEAD : BEHIND;
  resEl("resghost").hidden = resEl("resbeat").hidden = !first;
  resEl("ressaved").hidden = true;
  resName.value = `JUDGE ${resSaves + 1}`;
}

function results(s) {
  const r = s.race, panel = resEl("results");
  const done = !!r && r.state === "finished" && !!r.score;
  if (!done) resAt = null;
  else if (resAt === null) resShow(r);
  panel.hidden = !done || performance.now() - resAt < RES_DELAY_MS;
  // typing in the field skips the space = kill key, so it only takes keys while disarmed
  resName.disabled = !!s.armed || resSaved || panel.hidden;
  if (resName.disabled) resName.blur();
}
