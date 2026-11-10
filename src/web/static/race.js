// the plate seen from above, drawn in mm: origin at the centre, +x right, +y up
const TRAIL_S = 0.5;
const BOARD = "#F7F7F2";
const RIM = "#C4A77D";
const MARKER = "#1B1B1B";
const PATH = "#0072B2";
const GATE = "#4A5380";
const GATE_LIT = "#00795A";
const CROSS = "#FF4B3A";
const BALL = "#141414";
const SHINE = "#9AA5CC";
const GHOST = "#FF5FA2";
const GHOST_EDGE = "#D63A84";
const TICKS = "#5A6699";
const MARGIN_MM = 17;  // room round the plate for the rim and the scale
const TICK_MM = 10;
const NUM_FONT = "'Press Start 2P', monospace";
const GATE_R = 11 / 8;  // gate ring in ball radii
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

// a tick every 10 mm from the centre, along all four sides
function ticks(ctx, h) {
  const a = h + 5.5, b = h + 10.5;
  ctx.lineWidth = 0.6;
  ctx.strokeStyle = TICKS;
  ctx.beginPath();
  for (let v = -Math.floor(h / TICK_MM) * TICK_MM; v <= h; v += TICK_MM) {
    for (const e of [-1, 1]) {
      ctx.moveTo(v, e * a);
      ctx.lineTo(v, e * b);
      ctx.moveTo(e * a, v);
      ctx.lineTo(e * b, v);
    }
  }
  ctx.stroke();
}

function plate(ctx, h) {
  ctx.beginPath();
  ctx.roundRect(-h, -h, 2 * h, 2 * h, 4);
  ctx.fillStyle = BOARD;
  ctx.fill();
  ctx.lineWidth = 6;
  ctx.strokeStyle = RIM;
  ctx.stroke();
}

function walls(ctx, maze) {
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = 3.2;
  ctx.strokeStyle = MARKER;
  maze.walls.forEach((w) => polyline(ctx, w));
}

function route(ctx, maze) {
  ctx.lineCap = "butt";
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = PATH;
  ctx.setLineDash([3, 3]);
  polyline(ctx, maze.points);
  ctx.setLineDash([]);
}

// upright text centred on [x, y] mm
function label(ctx, [x, y], size, text, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(1, -1);
  ctx.font = `${size}px ${NUM_FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function gate(ctx, at, r, n, hit, next, last) {
  const edge = hit ? GATE_LIT : next ? PATH : GATE;
  circle(ctx, at, r);
  ctx.fillStyle = hit ? GATE_LIT : BOARD;
  ctx.fill();
  ctx.strokeStyle = edge;
  ctx.lineWidth = next ? 2.6 : 1.6;
  if (!hit) ctx.stroke();
  if (!last) return label(ctx, at, (r * 8) / 11, String(n + 1), hit ? "#FFFFFF" : edge);
  ctx.lineWidth = hit ? 2 : 1.6;
  circle(ctx, at, r + 3.5);
  ctx.stroke();
}

function gates(ctx, maze, ballR, race) {
  const r = ballR * GATE_R, finish = maze.checkpoints.length - 1, now = performance.now();
  const reached = race && race.state ? race.reached : 0;
  const live = race && (race.state === "countdown" || race.state === "running");
  const pulse = 0.6 + 0.4 * Math.cos((2 * Math.PI * now) / GATE_PULSE_MS);
  if (reached > gatesHit) gateHitAt = now;
  gatesHit = reached;
  maze.checkpoints.forEach((i, n) => {
    const next = live && n === reached;
    ctx.globalAlpha = next ? pulse : 1;
    gate(ctx, maze.points[i], r, n, n < reached, next, n === finish);
  });
  ctx.globalAlpha = 1;
  const since = now - gateHitAt;
  if (reached && since < GATE_FLASH_MS) {
    ctx.strokeStyle = GATE_LIT;
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

function ghost(ctx, at, r, now) {
  tail(ctx, ghostTrail, now, TRAIL_S, 1.6, GHOST_EDGE, 0.6);
  circle(ctx, at, r);
  ctx.fillStyle = GHOST;
  ctx.globalAlpha = 0.3;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = GHOST_EDGE;
  ctx.stroke();
}

// crosshair across the whole plate, a ring, then the ball itself
function ball(ctx, [x, y], r, h) {
  ctx.lineCap = "butt";
  ctx.lineWidth = 0.7;
  ctx.strokeStyle = CROSS;
  polyline(ctx, [[x, -h], [x, h]]);
  polyline(ctx, [[-h, y], [h, y]]);
  ctx.lineWidth = 1.4;
  circle(ctx, [x, y], r * 1.5);
  ctx.stroke();
  ctx.fillStyle = BALL;
  circle(ctx, [x, y], r);
  ctx.fill();
  ctx.fillStyle = SHINE;
  circle(ctx, [x - r * 0.31, y + r * 0.31], r * 0.275);
  ctx.fill();
}

function draw(ctx, cfg, maze, s) {
  const px = fit(ctx.canvas), h = cfg.plate_inner_mm / 2, r = cfg.ball_d_mm / 2;
  const k = px / (2 * (h + MARGIN_MM));
  const g = s.race && s.race.ghost_mm;
  keep(trail, s.ball.valid && [s.ball.x_mm, s.ball.y_mm], s.t_s, TRAIL_S);
  if (g) keep(ghostTrail, g, s.t_s, TRAIL_S); else ghostTrail.length = 0;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, px, px);
  ctx.setTransform(k, 0, 0, -k, px / 2, px / 2);
  ticks(ctx, h);
  plate(ctx, h);
  guardLine(ctx, s, h);
  if (maze) {
    if (maze.walls) walls(ctx, maze);
    route(ctx, maze);
    gates(ctx, maze, r, s.race);
  }
  if (g) ghost(ctx, g, r, s.t_s);
  if (s.choices.mode === "balance") reticle(ctx, s.target, r);
  tail(ctx, trail, s.t_s, TRAIL_S, r / 2, BALL, 0.25);
  // the player's ball goes on last so nothing drawn later can cover it
  if (s.ball.valid) ball(ctx, [s.ball.x_mm, s.ball.y_mm], r, h);
}
