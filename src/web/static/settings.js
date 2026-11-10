// the settings page: shows /settings.json and sends back only what was changed
const $ = id => document.getElementById(id);
const API = "/settings.json";
const UNITS = { us: "µs", s: "s", ms: "ms", mm: "mm", deg: "°", hz: "Hz", px: "px", fps: "fps", s2: "s²" };
const SECTIONS = [
  ["rig", "RIG AND SERVOS", "Read only here. Servo limits, safety times and the board link change in the settings file."],
  ["input", "HAND AND KEYS", "How your palm or the keyboard turns into tilt."],
  ["control", "CONTROL LOOP", "Gains and models behind balance, path and assist. They apply on the next arm."],
  ["game", "RACE RULES", "Countdown, checkpoints and when a run is called off."],
  ["vision", "BALL TRACKING AND CALIBRATION", "What counts as the ball, and how pixels become millimetres."],
];
// how you play: the group of buttons that edits each of these
const PICKS = { pickmode: "control.mode", pickhand: "input.hand_mode", pickmaze: "game.maze" };
const SAVED = "Saved to settings.local.toml. Camera, hand tracker and maze changes need a restart.";
let data = null, dirty = {};
const editors = [];

// "kd_deg_s_mm" -> ["KD", "°·s/mm"], "us_per_deg_x" -> ["US PER DEG X", "µs/°"]
function fieldName(key) {
  const words = key.split("_"), per = words.indexOf("per"), units = [];
  if (per > 0) return [words.join(" ").toUpperCase(), `${UNITS[words[per - 1]]}/${UNITS[words[per + 1]]}`];
  while (words.length > 1 && UNITS[words[words.length - 1]]) units.unshift(UNITS[words.pop()]);
  const top = units.slice(0, -1).join("·");
  return [words.join(" ").toUpperCase(), top ? `${top}/${units[units.length - 1]}` : units[0] || ""];
}

const shown = v => (typeof v === "string" ? v : JSON.stringify(v));

// what was typed as the kind of value the setting is, undefined when it does not read as one
function typed(text, old) {
  if (typeof old === "string") return text;
  if (typeof old === "number") return text.trim() !== "" && Number.isFinite(Number(text)) ? Number(text) : undefined;
  try {
    const v = JSON.parse(text);
    return Array.isArray(v) && v.length === old.length ? v : undefined;
  } catch {
    return undefined;
  }
}

function make(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const saved = path => { const [s, k] = path.split("."); return data.sections[s][k]; };
const now = path => (path in dirty ? dirty[path] : saved(path));
const locked = path => data.armed || data.locked.includes(path.split(".")[0]);

function edit(path, value) {
  if (value !== undefined && shown(value) === shown(saved(path))) delete dirty[path];
  else dirty[path] = value;
  paint();
}

function flip(path, name) {
  const b = make("button", "switch");
  b.type = "button";
  b.setAttribute("id", path);
  b.setAttribute("role", "switch");
  b.setAttribute("aria-label", name);
  b.append(make("span"));
  b.onclick = () => edit(path, !now(path));
  editors.push(() => { b.setAttribute("aria-checked", now(path)); b.disabled = locked(path); });
  return b;
}

function entry(path, name, unit) {
  const wrap = make("div", "value"), input = make("input");
  input.setAttribute("id", path);
  input.setAttribute("aria-label", name);
  input.setAttribute("autocomplete", "off");
  input.setAttribute("spellcheck", "false");
  input.value = shown(saved(path));
  input.oninput = () => edit(path, typed(input.value, saved(path)));
  wrap.append(input);
  if (unit) wrap.append(make("span", "dim", unit));
  editors.push(() => {
    input.disabled = locked(path);
    wrap.className = path in dirty && dirty[path] === undefined ? "value bad" : "value";
    if (!(path in dirty)) input.value = shown(saved(path));
  });
  return wrap;
}

function field(section, key) {
  const path = `${section}.${key}`, [name, unit] = fieldName(key), box = make("div", "field");
  box.append(make("span", "dim", name));
  box.append(typeof saved(path) === "boolean" ? flip(path, name) : entry(path, name, unit));
  box.append(make("span", "key", path));
  return box;
}

function build() {
  const elsewhere = [...Object.values(PICKS), "game.ghost"];
  $("jump").replaceChildren();
  $("sections").replaceChildren();
  editors.length = 0;
  for (const [id, title, about] of SECTIONS.filter(([s]) => data.sections[s])) {
    const link = make("a", "pill", title), part = make("section"), head = make("div", "head");
    const grid = make("div", "grid");
    link.href = `#${id}`;
    part.id = id;
    head.append(make("h2", "", title), make("span", "dim", about));
    for (const key of Object.keys(data.sections[id])) {
      if (!elsewhere.includes(`${id}.${key}`)) grid.append(field(id, key));
    }
    part.append(head, grid);
    $("jump").append(link);
    $("sections").append(part);
  }
}

function press(group, value, off) {
  for (const b of $(group).children) {
    b.setAttribute("aria-pressed", b.dataset.pick === value);
    if (off !== undefined) b.disabled = off(b);
  }
}

function paint() {
  const bad = Object.values(dirty).includes(undefined), any = Object.keys(dirty).length > 0;
  editors.forEach(fn => fn());
  for (const [group, path] of Object.entries(PICKS)) {
    const missing = b => group === "pickmaze" && !data.mazes.includes(b.dataset.pick);
    press(group, now(path), b => locked(path) || !b.dataset.pick || missing(b));
  }
  press("picksteer", data.choices.steer);
  press("picktarget", data.target);
  if ("guard_on" in data.sections.control) {
    $("pickguard").setAttribute("aria-checked", now("control.guard_on"));
    $("pickguard").disabled = locked("control.guard_on");
  }
  $("save").disabled = data.armed || bad || !any;
  if (data.armed) say("Armed: locked. Kill on the play screen first.", true);
}

function say(text, bad) {
  $("note").textContent = text;
  $("note").className = bad ? "red" : "dim";
}

function show(json, text) {
  data = json;
  dirty = {};
  build();
  say(text || "Every value from the settings files. Locked while armed.", false);
  paint();
}

const get = () => fetch(API).then(r => r.json());

async function save() {
  const headers = { "Content-Type": "application/json" };
  const r = await fetch(API, { method: "POST", headers, body: JSON.stringify(dirty) });
  const json = await r.json();
  if (r.ok) show(json, SAVED); else say(json.error, true);
}

for (const [group, path] of Object.entries(PICKS)) {
  for (const b of $(group).children) b.onclick = () => edit(path, b.dataset.pick);
}
$("pickguard").onclick = () => edit("control.guard_on", !now("control.guard_on"));
$("save").onclick = save;
$("revert").onclick = () => get().then(json => show(json));
// the lock follows the plate without wiping what is being typed
setInterval(() => get().then(json => {
  if (!data || json.armed === data.armed) return;
  data.armed = json.armed;
  say("Every value from the settings files. Locked while armed.", false);
  paint();
}).catch(() => {}), 1000);
get().then(json => show(json)).catch(e => say(`load failed: ${e.message}`, true));
