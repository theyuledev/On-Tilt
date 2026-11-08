// what the loop is steering by, in intent units: +x right, +y up, half-width = 1
const PAD_FILL = 0.84;  // the rest of the half-width is the hold-still ring

function padGrid(ctx, deadZone) {
  ctx.lineWidth = 0.02;
  ctx.strokeStyle = "#666";
  polyline(ctx, [[-1, 0], [1, 0]]);
  polyline(ctx, [[0, -1], [0, 1]]);
  circle(ctx, [0, 0], deadZone);
  ctx.stroke();
}

function padDot(ctx, intent, deadZone) {
  circle(ctx, [intent.x, intent.y], 0.06);
  ctx.lineWidth = 0.03;
  ctx.strokeStyle = ctx.fillStyle = "#ff3";
  if (Math.hypot(intent.x, intent.y) >= deadZone) ctx.fill(); else ctx.stroke();
}

function padRing(ctx, h, progress) {
  ctx.beginPath();
  ctx.arc(h, h, h * 0.96, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * Math.min(1, progress));
  ctx.lineWidth = h * 0.06;
  ctx.strokeStyle = "#ffb300";
  ctx.stroke();
}

function drawPad(ctx, cfg, s) {
  const px = fit(ctx.canvas), h = px / 2, k = h * PAD_FILL;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, px, px);
  ctx.setTransform(k, 0, 0, -k, h, h);
  padGrid(ctx, cfg.dead_zone);
  padDot(ctx, s.intent, cfg.dead_zone);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (s.hand_progress !== undefined) padRing(ctx, h, s.hand_progress);
  if (s.hand_seen && s.hand_calibrating) {
    ctx.font = `${Math.round(px / 9)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillStyle = "#ffb300";
    ctx.fillText("hold still", h, px * 0.8);
  }
}
