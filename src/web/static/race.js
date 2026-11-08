// the plate seen from above, drawn in mm: origin at the centre, +x right, +y up
const TRAIL_S = 0.5;
const PLAYER = "#fff";
const GHOST = "#CC79A7";
const GHOST_RING_PX = 4;
const GHOST_TRAIL_PX = 3;
const TAG_FONT = "700 20px system-ui, sans-serif";
const WALL = "#bbb";
const GATE = "#777";
const GATE_LIT = "#009E73";
const GATE_PULSE_MS = 600;
const GATE_FLASH_MS = 120;
const trail = [];
const ghostTrail = [];
let gatesHit = 0, gateHitAt = -Infinity;

function fit(canvas) {
  const px = Math.round(canvas.clientWidth * devicePixelRatio);
  if (canvas.width !== px) canvas.width = canvas.height = px;
  return px;
}

// pt is [x, y] in mm or falsy for "nothing to add this frame"
function keep(buf, pt, now, keepS) {
  if (pt) buf.push({ x: pt[0], y: pt[1], t: now });
  while (buf.length && now - buf[0].t > keepS) buf.shift();
}

function fmtLap(s) {
  const h = Math.round(s * 100), pad = (n) => String(n).padStart(2, "0");
  return `${pad(Math.floor(h / 6000))}:${pad(Math.floor((h % 6000) / 100))}.${pad(h % 100)}`;
}

function polyline(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

function circle(ctx, [x, y], r) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
}

function plate(ctx, mm) {
  ctx.fillStyle = "#1c1c1c";
  ctx.fillRect(-mm / 2, -mm / 2, mm, mm);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ccc";
  ctx.strokeRect(-mm / 2, -mm / 2, mm, mm);
}

function walls(ctx, maze) {
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = 2;
  ctx.strokeStyle = WALL;
  maze.walls.forEach((w) => polyline(ctx, w));
}

function ribbon(ctx, maze, w) {
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = w;
  ctx.strokeStyle = "#333";
  polyline(ctx, maze.points);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#555";
  polyline(ctx, maze.points);
}

function gates(ctx, maze, r, race) {
  const finish = maze.checkpoints.length - 1, now = performance.now();
  const reached = race && race.state ? race.reached : 0;
  const live = race && (race.state === "countdown" || race.state === "running");
  const pulse = 0.6 + 0.4 * Math.cos((2 * Math.PI * now) / GATE_PULSE_MS);
  if (reached > gatesHit) gateHitAt = now;
  gatesHit = reached;
  maze.checkpoints.forEach((i, n) => {
    ctx.strokeStyle = n < reached ? GATE_LIT : GATE;
    ctx.globalAlpha = live && n === reached ? pulse : 1;
    ctx.lineWidth = n === finish ? 3 : 2;
    circle(ctx, maze.points[i], r);
    ctx.stroke();
    if (n === finish) { circle(ctx, maze.points[i], r + 6); ctx.stroke(); }
  });
  ctx.globalAlpha = 1;
  const since = now - gateHitAt;
  if (reached && since < GATE_FLASH_MS) {
    ctx.strokeStyle = PLAYER;
    ctx.lineWidth = 3;
    circle(ctx, maze.points[maze.checkpoints[reached - 1]], r * (1 + since / GATE_FLASH_MS));
    ctx.stroke();
  }
  ctx.fillStyle = GATE;
  circle(ctx, maze.start, 4);
  ctx.fill();
}

function tail(ctx, buf, now, keepS, w, color, alpha) {
  ctx.lineWidth = w;
  ctx.lineCap = "round";
  ctx.strokeStyle = color;
  for (let i = 1; i < buf.length; i++) {
    const a = buf[i - 1], b = buf[i];
    ctx.globalAlpha = alpha * (1 - (now - b.t) / keepS);
    polyline(ctx, [[a.x, a.y], [b.x, b.y]]);
  }
  ctx.globalAlpha = 1;
}

// k is mm per device px; the text is drawn upright in device px, bottom-left at [x, y] mm
function tag(ctx, [x, y], k, text) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(k, -k);
  ctx.font = TAG_FONT;
  ctx.textBaseline = "bottom";
  ctx.fillStyle = GHOST;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function ghost(ctx, race, r, k, now) {
  const g = race.ghost_mm;
  tail(ctx, ghostTrail, now, TRAIL_S, GHOST_TRAIL_PX * k, GHOST, 0.5);
  circle(ctx, g, r);
  ctx.fillStyle = GHOST;
  ctx.globalAlpha = 0.25;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = GHOST_RING_PX * k;
  ctx.strokeStyle = GHOST;
  ctx.stroke();
  if (typeof race.ghost_lap_s === "number") tag(ctx, [g[0] + r, g[1] + r], k, `BEST ${fmtLap(race.ghost_lap_s)}`);
}

function draw(ctx, cfg, maze, s) {
  const px = fit(ctx.canvas), mm = cfg.plate_inner_mm, r = cfg.ball_d_mm / 2, k = mm / px;
  const g = s.race && s.race.ghost_mm;
  keep(trail, s.ball.valid && [s.ball.x_mm, s.ball.y_mm], s.t_s, TRAIL_S);
  if (g) keep(ghostTrail, g, s.t_s, TRAIL_S); else ghostTrail.length = 0;
  ctx.setTransform(px / mm, 0, 0, -px / mm, px / 2, px / 2);
  plate(ctx, mm);
  if (maze) {
    if (maze.walls) walls(ctx, maze);
    ribbon(ctx, maze, cfg.ball_d_mm);
    gates(ctx, maze, r, s.race);
  }
  tail(ctx, trail, s.t_s, TRAIL_S, r / 2, PLAYER, 1);
  if (g) ghost(ctx, s.race, r, k, s.t_s);
  // the player's ball goes on last so nothing drawn later can cover it
  if (!s.ball.valid) return;
  ctx.fillStyle = PLAYER;
  circle(ctx, [s.ball.x_mm, s.ball.y_mm], r);
  ctx.fill();
}
