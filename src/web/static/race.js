// the plate seen from above, drawn in mm: origin at the centre, +x right, +y up
const trail = [];

function fit(canvas) {
  const px = Math.round(canvas.clientWidth * devicePixelRatio);
  if (canvas.width !== px) canvas.width = canvas.height = px;
  return px;
}

function keep(ball, now, keepS) {
  if (ball.valid) trail.push({ x: ball.x_mm, y: ball.y_mm, t: now });
  while (trail.length && now - trail[0].t > keepS) trail.shift();
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

function ribbon(ctx, maze, w) {
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = w;
  ctx.strokeStyle = "#333";
  polyline(ctx, maze.points);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#555";
  polyline(ctx, maze.points);
}

function gates(ctx, maze, r) {
  const finish = maze.checkpoints.length - 1;
  ctx.strokeStyle = ctx.fillStyle = "#777";
  maze.checkpoints.forEach((i, n) => {
    ctx.lineWidth = n === finish ? 3 : 2;
    circle(ctx, maze.points[i], r);
    ctx.stroke();
    if (n === finish) { circle(ctx, maze.points[i], r + 6); ctx.stroke(); }
  });
  circle(ctx, maze.start, 4);
  ctx.fill();
}

function tail(ctx, now, keepS, w) {
  ctx.lineWidth = w;
  ctx.lineCap = "round";
  for (let i = 1; i < trail.length; i++) {
    const a = trail[i - 1], b = trail[i];
    ctx.strokeStyle = `rgba(255, 140, 26, ${1 - (now - b.t) / keepS})`;
    polyline(ctx, [[a.x, a.y], [b.x, b.y]]);
  }
}

function draw(ctx, cfg, maze, s) {
  const px = fit(ctx.canvas), mm = cfg.plate_inner_mm, r = cfg.ball_d_mm / 2;
  keep(s.ball, s.t_s, cfg.trail_s);
  ctx.setTransform(px / mm, 0, 0, -px / mm, px / 2, px / 2);
  plate(ctx, mm);
  ribbon(ctx, maze, cfg.ball_d_mm);
  gates(ctx, maze, r);
  tail(ctx, s.t_s, cfg.trail_s, r / 2);
  // the player's ball goes on last so nothing drawn later can cover it
  if (!s.ball.valid) return;
  ctx.fillStyle = "#ff8c1a";
  circle(ctx, [s.ball.x_mm, s.ball.y_mm], r);
  ctx.fill();
}
