// what the loop is steering by, in intent units: +x right, +y up, the rim is 1
const PAD_VIEW = 1.1;
const PAD_DOT = "#FFD23F";

// hold still progress, clockwise from the top
function padRing(ctx, progress) {
  ctx.beginPath();
  ctx.arc(0, 0, 1, Math.PI / 2, Math.PI / 2 - 2 * Math.PI * Math.min(1, progress), true);
  ctx.lineWidth = 0.1;
  ctx.strokeStyle = PAD_DOT;
  ctx.stroke();
}

function padDot(ctx, intent, deadZone) {
  const d = Math.hypot(intent.x, intent.y), out = Math.max(1, d);
  circle(ctx, [intent.x / out, intent.y / out], d >= deadZone ? 0.14 : 0.12);
  ctx.fillStyle = PAD_DOT;
  ctx.fill();
}

function drawPad(ctx, cfg, s) {
  const px = fit(ctx.canvas), k = px / (2 * PAD_VIEW);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, px, px);
  ctx.setTransform(k, 0, 0, -k, px / 2, px / 2);
  circle(ctx, [0, 0], 1);
  ctx.fillStyle = "#0B0E1E";
  ctx.fill();
  ctx.lineWidth = 0.05;
  ctx.strokeStyle = "#5A6699";
  ctx.stroke();
  ctx.lineWidth = 0.04;
  ctx.strokeStyle = "#2B3564";
  polyline(ctx, [[-1, 0], [1, 0]]);
  polyline(ctx, [[0, -1], [0, 1]]);
  if (s.hand_progress !== undefined) padRing(ctx, s.hand_progress);
  padDot(ctx, s.intent, cfg.dead_zone);
}
