// the plate from above in perspective, unit coords: +x right, +y away, half-width = 1
const TILT_SHOW = 8;  // the real plate barely tilts, so the picture exaggerates it
const BOARD_EYE = 4;  // eye height over the plate, in half-widths
const BOARD_FILL = 0.3;  // plate half-width as a share of the canvas height
const CORNERS = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
const CMD_FILL = "#2a2a2a";
const REAL = "#56B4E9";
const GUARD = "#E69F00";
const HAND_LOOP_MS = 4000;
const HAND_SWAY_DEG = 25;
const HAND_POLY = [
  [-0.3, -0.6], [-0.32, -0.25], [-0.58, 0], [-0.6, 0.12], [-0.48, 0.12], [-0.32, 0.02],
  [-0.32, 0.6], [-0.18, 0.6], [-0.18, 0.2], [-0.15, 0.2], [-0.15, 0.68], [-0.01, 0.68],
  [-0.01, 0.2], [0.02, 0.2], [0.02, 0.6], [0.16, 0.6], [0.16, 0.2], [0.19, 0.2],
  [0.19, 0.45], [0.32, 0.45], [0.3, -0.6],
];

const clamp1 = v => Math.max(-1, Math.min(1, Number(v) || 0));

// +tx lowers the +x edge and +ty the +y edge; lower means further from the eye, so smaller
function boardPt(x, y, tx, ty, k) {
  const a = (tx * k * Math.PI) / 180, b = (ty * k * Math.PI) / 180;
  const s = BOARD_EYE / (BOARD_EYE + x * Math.sin(a) + y * Math.sin(b));
  return [x * Math.cos(a) * s, y * Math.cos(b) * s];
}

function boardQuad(tx, ty, k) {
  return CORNERS.map(([x, y]) => boardPt(x, y, tx, ty, k));
}

// the one place that knows the snapshot's guard field
function guardOf(s) {
  const g = s.guard;
  if (!g || !Array.isArray(g.w) || !Array.isArray(g.side)) return null;
  return { w: g.w.map(v => Number(v) || 0), side: g.side.map(v => Math.sign(v)) };
}

function glowAlpha(w) {
  return w < 0.05 ? 0 : Math.max(0.15, Math.min(0.6, w));
}

function ghostOn(s, cfg) {
  return !!cfg.hand && (!s.hand_seen || !!s.hand_calibrating);
}

function closed(ctx, pts) {
  polyline(ctx, [...pts, pts[0]]);
}

function glow(ctx, q, g) {
  ctx.lineWidth = 0.12;
  ctx.strokeStyle = GUARD;
  [0, 1].forEach(axis => {
    const edge = q.filter((_, i) => CORNERS[i][axis] === g.side[axis]), a = glowAlpha(g.w[axis]);
    if (!a || edge.length !== 2) return;
    ctx.globalAlpha = a;
    polyline(ctx, edge);
  });
  ctx.globalAlpha = 1;
}

// left, right, away, toward, one leg each per loop
function ghostHand(ctx, now) {
  const u = (now % HAND_LOOP_MS) / (HAND_LOOP_MS / 4), leg = Math.floor(u);
  const d = HAND_SWAY_DEG * Math.sin(Math.PI * (u - leg));
  const [tx, ty] = [[-d, 0], [d, 0], [0, d], [0, -d]][leg];
  ctx.globalAlpha = 0.4;
  ctx.fillStyle = ctx.strokeStyle = "#fff";
  ctx.lineWidth = 0.02;
  closed(ctx, HAND_POLY.map(([x, y]) => boardPt(x, y, tx, ty, 1)));
  ctx.fill();
  ctx.globalAlpha = 1;
}

function plateTop(ctx, s, cfg) {
  const p = s.plate_state, b = s.ball;
  const cmd = boardQuad(p.sent_x_deg, p.sent_y_deg, TILT_SHOW);
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = 0.02;
  ctx.fillStyle = CMD_FILL;
  ctx.strokeStyle = "#666";
  closed(ctx, cmd);
  ctx.fill();
  const g = guardOf(s);
  if (g) glow(ctx, cmd, g);
  ctx.lineWidth = 0.03;
  ctx.strokeStyle = REAL;
  closed(ctx, boardQuad(p.est_x_deg, p.est_y_deg, TILT_SHOW));
  if (!b.valid) return;
  const half = cfg.plate_inner_mm / 2;
  const at = boardPt(b.x_mm / half, b.y_mm / half, p.est_x_deg, p.est_y_deg, TILT_SHOW);
  circle(ctx, at, cfg.ball_d_mm / cfg.plate_inner_mm);
  ctx.fillStyle = "#fff";
  ctx.fill();
}

// each pair is x then y, drawn from the middle of its track
function bars(ctx, w, h, pairs) {
  const cw = w / 3, half = cw * 0.38, bh = Math.max(2, Math.round(h * 0.035));
  ctx.font = `${Math.round(h * 0.07)}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  pairs.forEach(([label, v, color], i) => {
    const cx = cw * (i + 0.5);
    ctx.fillStyle = "#aaa";
    ctx.fillText(label, cx, h * 0.86);
    v.forEach((u, j) => {
      const y = h * (0.9 + 0.055 * j);
      ctx.fillStyle = "#333";
      ctx.fillRect(cx - half, y, 2 * half, bh);
      ctx.fillStyle = color;
      ctx.fillRect(cx, y, half * clamp1(u), bh);
    });
  });
}

function board(s, cfg) {
  const c = $("tiltboard"), ctx = c.getContext("2d"), on = ghostOn(s, cfg);
  const w = fit(c), h = Math.round(w * 0.75), k = h * BOARD_FILL;
  if (c.height !== h) c.height = h;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.setTransform(k, 0, 0, -k, w / 2, h * 0.4);
  plateTop(ctx, s, cfg);
  if (on) ghostHand(ctx, performance.now());
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const p = s.plate_state, f = cfg.tilt_full_deg;
  bars(ctx, w, h, [
    ["hand", [s.intent.x, s.intent.y], "#ff3"],
    ["cmd", [p.sent_x_deg / f, p.sent_y_deg / f], "#ccc"],
    ["real", [p.est_x_deg / f, p.est_y_deg / f], REAL],
  ]);
  $("handhint").hidden = !on;
}
