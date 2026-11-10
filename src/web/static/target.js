// balance target: the reticle on the board, the click that moves it and the side readouts
// (uses race.js circle / polyline / MARGIN_MM / MARKER / PATH and app.js $)

// u, v run 0..1 across the board canvas from its top left corner
function plateMm(u, v, cfg) {
  const half = cfg.plate_inner_mm / 2 + MARGIN_MM;
  return [(2 * u - 1) * half, (1 - 2 * v) * half];
}

function distMm(s) {
  const b = s.ball, [x, y] = s.target.mm;
  return b.valid ? Math.hypot(b.x_mm - x, b.y_mm - y) : null;
}

// r is the ball radius, the rings scale with it
function reticle(ctx, t, r) {
  const [x, y] = t.mm, a = 1.75 * r, b = 4 * r;
  if (t.src === "marker" && t.found) {
    ctx.lineCap = "round";
    ctx.lineWidth = 3.2;
    ctx.strokeStyle = MARKER;
    polyline(ctx, [[x - r, y - r], [x + r, y + r]]);
    polyline(ctx, [[x - r, y + r], [x + r, y - r]]);
  }
  ctx.lineCap = "butt";
  ctx.strokeStyle = PATH;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  circle(ctx, [x, y], 3.125 * r);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 1.6;
  circle(ctx, [x, y], 1.25 * r);
  ctx.stroke();
  ctx.lineWidth = 1.2;
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    polyline(ctx, [[x + dx * a, y + dy * a], [x + dx * b, y + dy * b]]);
  }
}

function balanceReadouts(s) {
  const b = s.ball, p = s.plate_state, d = distMm(s);
  $("timelabel").textContent = "DISTANCE FROM TARGET";
  $("timer").className = "free";
  $("timer").textContent = d === null ? "--.- MM" : `${d.toFixed(1)} MM`;
  $("speed").textContent = `${b.valid ? Math.round(Math.hypot(b.vx_mm_s, b.vy_mm_s)) : 0}`;
  $("tiltnow").textContent = `${Math.hypot(p.sent_x_deg, p.sent_y_deg).toFixed(2)}°`;
}
