// name entry in #side after a finish; app.js calls results(s) each snapshot (uses its pending)
const RES_DELAY_MS = 1000;  // the FINISH pop over the board shows first
const resEl = id => document.getElementById(id);
const resName = resEl("resname");
let resAt = null, resSaves = 0, resSaved = false;

function resSave() {
  const name = resName.value.trim();
  if (!name) return;
  pending.save = name;
  resSaves += 1;
  resSaved = true;
  resName.blur();
  resEl("ressaved").hidden = false;
}

resName.addEventListener("keydown", e => {
  if (e.code === "Enter" || e.code === "NumpadEnter") resSave();
});
resEl("ressave").onclick = e => { resSave(); e.currentTarget.blur(); };

function resShow(r) {
  resAt = performance.now();
  resSaved = false;
  resEl("respen").textContent = `Penalty +${r.score.penalty_s.toFixed(1)} s.`;
  resEl("ressaved").hidden = true;
  resName.value = `JUDGE ${resSaves + 1}`;
}

function results(s) {
  const r = s.race, panel = resEl("results");
  const done = !!r && r.state === "finished" && !!r.score;
  if (!done) resAt = null;
  else if (resAt === null) resShow(r);
  panel.hidden = !done || performance.now() - resAt < RES_DELAY_MS;
  resEl("choices").hidden = !panel.hidden;
  // typing in the field skips the space = kill key, so it only takes keys while disarmed
  resName.disabled = resEl("ressave").disabled = !!s.armed || resSaved || panel.hidden;
  if (resName.disabled) resName.blur();
  resEl("resarm").hidden = panel.hidden || !s.armed || resSaved;
}
