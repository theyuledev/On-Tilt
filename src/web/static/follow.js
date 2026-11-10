// follow cam: the board drawing zoomed on the ball and tilted back in css, the marker walls
// stood up, a tag with the ball position and a whole-board map in the corner
// (uses race.js walls / plate / circle / fit, gates.js label, hud.js signed)
const FOLLOW_ZOOM = 2.3;
const FOLLOW_SCALE = 1.32;  // the css scale on top of the zoom, together the 3x
const LIFTS_MM = [1.5, 3, 4.5];  // copies of the strokes stacked up the screen
const WALL_TOP_MM = 6;
const WALL_TOP = "#6E6E69";

// the window sits a little ahead of the ball so the tag under it stays in view
function followLook(eye, r) {
  return { x: eye[0], y: eye[1] + 0.8 * r, zoom: FOLLOW_ZOOM, tall: true };
}

function tallWalls(ctx, maze) {
  walls(ctx, maze);
  for (const lift of [...LIFTS_MM, WALL_TOP_MM]) {
    ctx.save();
    ctx.translate(0, lift);
    walls(ctx, maze, lift === WALL_TOP_MM ? WALL_TOP : MARKER);
    ctx.restore();
  }
}

function shadow(ctx, [x, y], r) {
  ctx.fillStyle = "#000000";
  ctx.globalAlpha = 0.25;
  ctx.beginPath();
  ctx.ellipse(x + 0.275 * r, y - 0.39 * r, 1.06 * r, 0.875 * r, 0, 0, 2 * Math.PI);
  ctx.fill();
  ctx.globalAlpha = 1;
}

function posTag(ctx, [x, y], r) {
  const w = 5.9 * r, h = 1.25 * r, at = [x, y - 2.45 * r];
  ctx.fillStyle = "#0B0E1E";
  ctx.beginPath();
  ctx.roundRect(at[0] - w / 2, at[1] - h / 2, w, h, h / 2);
  ctx.fill();
  label(ctx, at, 0.94 * r, `X${signed(x, 1)} Y${signed(y, 1)}`, "#FF7A6B", "'VT323', monospace");
}

// the part of the board the follow cam shows, [x, y, w, h] in mm: the css scale narrows the
// window and the tilt shows a bit more of the near side. kept on the plate
function followBox(look, h) {
  const half = (h + MARGIN_MM) / look.zoom / FOLLOW_SCALE, mid = look.y - 0.16 * half;
  const lo = v => Math.max(-h, v - half), hi = v => Math.min(h, v + half);
  return [lo(look.x), lo(mid), hi(look.x) - lo(look.x), hi(mid) - lo(mid)];
}

// the whole board with that part boxed in red
function minimap(ctx, cfg, maze, s, look) {
  const px = fit(ctx.canvas), h = cfg.plate_inner_mm / 2, k = px / (2 * (h + 3));
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, px, px);
  ctx.setTransform(k, 0, 0, -k, px / 2, px / 2);
  plate(ctx, h);
  if (maze && maze.walls) walls(ctx, maze, MARKER, 4);
  ctx.strokeStyle = CROSS;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.roundRect(...followBox(look, h), 6);
  ctx.stroke();
  if (!s.ball.valid) return;
  ctx.fillStyle = CROSS;
  circle(ctx, [s.ball.x_mm, s.ball.y_mm], 1.125 * (cfg.ball_d_mm / 2));
  ctx.fill();
}
