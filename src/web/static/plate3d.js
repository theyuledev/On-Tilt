// the plate as a slab in perspective for the 3D view: plate mm in, canvas px out, the camera
// comes from cam.js (uses race.js fit and colours, gates.js GATE / GATE_LIT / GATE_R / NUM_FONT)
const EYE_PX = 1300;  // eye distance in scene px
const PLATE_PX = 270;  // the plate's half width in scene px, whatever it measures in mm
const SCENE_PX = 900;  // scene px across the canvas
const HORIZON = 0.54;  // the point the camera looks at sits this far down the canvas
const WALL_PX = 24, SLAB_PX = 30, LABEL_PX = 56, TAG_PX = 62;  // heights over the plate
const TILT_SHOW = 3;  // the real plate barely tilts, so the picture exaggerates it
const SLAB = ["#B4966B", "#BC9E73", "#C4A77D", "#BC9E73"];
const RING = Array.from({ length: 25 }, (_, i) => [Math.cos((i * Math.PI) / 12), Math.sin((i * Math.PI) / 12)]);

// p(x, y, z) -> [canvas x, canvas y, depth, px per scene px]; x, y plate mm, z scene px up
function projector(cam, tilt, half, px) {
  const u = PLATE_PX / half, f = px / SCENE_PX, rad = Math.PI / 180;
  const cy = Math.cos(cam.yaw * rad), sy = Math.sin(cam.yaw * rad);
  const cp = Math.cos(cam.pitch * rad), sp = Math.sin(cam.pitch * rad);
  // a positive tilt lowers that axis's + edge
  const lx = Math.sin(TILT_SHOW * tilt[0] * rad), ly = Math.sin(TILT_SHOW * tilt[1] * rad);
  return (x, y, z = 0) => {
    const h = z - (x * lx + y * ly) * u, dx = (x - cam.cx) * u, dy = (y - cam.cy) * u;
    const X = dx * cy - dy * sy, Y = dx * sy + dy * cy;
    const deep = (Y * sp - h * cp) * cam.zoom;
    const k = (f * cam.zoom * EYE_PX) / Math.max(EYE_PX + deep, 50);
    return [px / 2 + X * k, HORIZON * px - (Y * cp + h * sp) * k, deep, k];
  };
}

function trace(ctx, P, pts, close) {
  ctx.beginPath();
  pts.forEach(([x, y, z], i) => {
    const [sx, sy] = P(x, y, z);
    if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy);
  });
  if (close) ctx.closePath();
}

function slab(ctx, P, h) {
  const c = [[-h, h], [h, h], [h, -h], [-h, -h]];
  c.forEach(([ax, ay], i) => {
    const [bx, by] = c[(i + 1) % 4];
    ctx.fillStyle = SLAB[i];
    trace(ctx, P, [[ax, ay, 0], [bx, by, 0], [bx, by, -SLAB_PX], [ax, ay, -SLAB_PX]], true);
    ctx.fill();
  });
  ctx.fillStyle = BOARD;
  trace(ctx, P, c.map(([x, y]) => [x, y, 0]), true);
  ctx.fill();
}

function ring3(ctx, P, [x, y], r, z = 1) {
  trace(ctx, P, RING.map(([c, s]) => [x + r * c, y + r * s, z]), true);
}

// what lies flat on the plate: the path, the marker strokes and the gate rings
function flat(ctx, P, maze, r, race, k) {
  const reached = race && race.state ? race.reached : 0;
  ctx.strokeStyle = PATH;
  ctx.lineWidth = 2 * k;
  ctx.setLineDash([6 * k, 6 * k]);
  trace(ctx, P, maze.points.map(([x, y]) => [x, y, 1]));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = MARKER;
  ctx.lineWidth = 7 * k;
  (maze.walls || []).forEach(w => { trace(ctx, P, w.map(([x, y]) => [x, y, 1])); ctx.stroke(); });
  ctx.lineWidth = 3 * k;
  maze.checkpoints.forEach((i, n) => {
    ctx.strokeStyle = n < reached ? GATE_LIT : n === reached ? PATH : GATE;
    ring3(ctx, P, maze.points[i], r * GATE_R);
    ctx.stroke();
  });
}

// things that stand up from the plate, each with the depth it is sorted by
function barriers(ctx, P, maze) {
  return (maze.walls || []).flatMap(w => w.slice(1).map(([bx, by], i) => {
    const [ax, ay] = w[i];
    return [P((ax + bx) / 2, (ay + by) / 2)[2], () => {
      ctx.fillStyle = "rgba(27, 27, 27, 0.9)";
      trace(ctx, P, [[ax, ay, 0], [bx, by, 0], [bx, by, WALL_PX], [ax, ay, WALL_PX]], true);
      ctx.fill();
      ctx.strokeStyle = CROSS;
      ctx.lineWidth = 2 * P(ax, ay, WALL_PX)[3];
      trace(ctx, P, [[ax, ay, WALL_PX], [bx, by, WALL_PX]]);
      ctx.stroke();
    }];
  }));
}

function gateSigns(ctx, P, maze, race) {
  const reached = race && race.state ? race.reached : 0, last = maze.checkpoints.length - 1;
  return maze.checkpoints.map((i, n) => {
    const [x, y, deep, k] = P(...maze.points[i], LABEL_PX);
    return [deep, () => {
      ctx.font = `${12 * k}px ${NUM_FONT}`;
      ctx.textAlign = "center";
      ctx.fillStyle = n < reached ? GATE_LIT : n === reached ? PATH : GATE;
      ctx.fillText(n === last ? "FIN" : String(n + 1), x, y);
    }];
  });
}

function orb(ctx, P, [x, y], r, u, paint) {
  const [sx, sy, deep, k] = P(x, y, r * u);
  return [deep, () => { ctx.beginPath(); ctx.arc(sx, sy, r * u * k, 0, 2 * Math.PI); paint(sx, sy, r * u * k); }];
}

function ballPaint(ctx) {
  return (sx, sy, rad) => {
    const shade = ctx.createRadialGradient(sx - 0.3 * rad, sy - 0.4 * rad, 0, sx - 0.3 * rad, sy - 0.4 * rad, 1.5 * rad);
    shade.addColorStop(0, SHINE);
    shade.addColorStop(0.3, "#323E66");
    shade.addColorStop(0.75, BALL);
    ctx.fillStyle = shade;
    ctx.fill();
  };
}

function ghostPaint(ctx) {
  return (sx, sy, rad) => {
    ctx.fillStyle = GHOST;
    ctx.globalAlpha = 0.3;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = GHOST_EDGE;
    ctx.lineWidth = rad / 5;
    ctx.stroke();
  };
}

// the ball's place on the plate: crosshair, shadow and a tag that keeps its size on screen
function mark(ctx, P, [x, y], h, r, k, f) {
  ctx.strokeStyle = CROSS;
  ctx.lineWidth = 1.5 * k;
  trace(ctx, P, [[x, -h, 1], [x, h, 1]]);
  ctx.stroke();
  trace(ctx, P, [[-h, y, 1], [h, y, 1]]);
  ctx.stroke();
  ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
  ring3(ctx, P, [x + 0.3 * r, y - 0.45 * r], 1.1 * r);
  ctx.fill();
  const [sx, sy, deep] = P(x, y, TAG_PX), text = `x ${signed(x, 1)}  y ${signed(y, 1)} mm`;
  return [deep - 1, () => {
    ctx.font = `${23 * f}px 'VT323', monospace`;
    ctx.textAlign = "center";
    const w = ctx.measureText(text).width + 24 * f;
    ctx.fillStyle = "#0B0E1E";
    ctx.beginPath();
    ctx.roundRect(sx - w / 2, sy - 20 * f, w, 28 * f, 14 * f);
    ctx.fill();
    ctx.fillStyle = "#FF7A6B";
    ctx.fillText(text, sx, sy);
  }];
}

function draw3d(ctx, cfg, maze, s, cam) {
  const px = fit(ctx.canvas), h = cfg.plate_inner_mm / 2, r = cfg.ball_d_mm / 2, u = PLATE_PX / h;
  const tilt = [s.plate_state.sent_x_deg, s.plate_state.sent_y_deg], P = projector(cam, tilt, h, px);
  const k = P(cam.cx, cam.cy)[3], at = [s.ball.x_mm, s.ball.y_mm], g = s.race && s.race.ghost_mm;
  const up = [];
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, px, px);
  ctx.lineJoin = ctx.lineCap = "round";
  slab(ctx, P, h);
  if (maze) {
    flat(ctx, P, maze, r, s.race, k);
    up.push(...barriers(ctx, P, maze), ...gateSigns(ctx, P, maze, s.race));
  }
  if (s.choices.mode === "balance") {
    ctx.strokeStyle = PATH;
    ctx.lineWidth = 3 * k;
    ring3(ctx, P, s.target.mm, 1.25 * r);
    ctx.stroke();
  }
  if (g) up.push(orb(ctx, P, g, r, u, ghostPaint(ctx)));
  if (s.ball.valid) up.push(mark(ctx, P, at, h, r, k, px / SCENE_PX), orb(ctx, P, at, r, u, ballPaint(ctx)));
  // far things first so near ones cover them
  up.sort((a, b) => b[0] - a[0]).forEach(([, paint]) => paint());
}
