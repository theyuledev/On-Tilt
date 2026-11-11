// the plate as a tray in perspective for the 3D view, drawn through lens.js with the camera from
// cam.js (uses race.js fit and colours, gates.js GATE / GATE_LIT / GATE_R / NUM_FONT, follow.js
// WALL_TOP, hud.js signed)
const WALL_PX = 10;  // how high a marker stroke stands, well under the top of the ball
const RIM_PX = 10;  // how high the rim stands round the board
const SLAB_PX = 18;  // how thick the tray is under the board
const LABEL_PX = 56;  // how high a gate's number hangs over the plate
const TAG_MIN_PX = 18;  // the ball's tag text never gets smaller than this on screen
const SHADOW = { dark: 0.45, blur: 40, drop: 18 };  // the tray's shadow on the ground, scene px
const SLAB = ["#B4966B", "#BC9E73", "#C4A77D", "#BC9E73"];
const WALL_SIDE = "rgba(27, 27, 27, 0.9)";
const RING = Array.from({ length: 25 }, (_, i) => [Math.cos((i * Math.PI) / 12), Math.sin((i * Math.PI) / 12)]);

function trace(ctx, P, pts, close) {
  ctx.beginPath();
  pts.forEach(([x, y, z], i) => {
    const [sx, sy] = P(x, y, z);
    if (i) ctx.lineTo(sx, sy); else ctx.moveTo(sx, sy);
  });
  if (close) ctx.closePath();
}

// the board, the slab under it and a soft shadow on the ground; f is canvas px per scene px
function tray(ctx, P, h, f) {
  const c = [[-h, h], [h, h], [h, -h], [-h, -h]];
  ctx.save();
  ctx.shadowColor = `rgba(0, 0, 0, ${SHADOW.dark})`;
  ctx.shadowBlur = SHADOW.blur * f;
  ctx.shadowOffsetY = SHADOW.drop * f;
  ctx.fillStyle = SLAB[0];
  trace(ctx, P, c.map(([x, y]) => [x, y, -SLAB_PX]), true);
  ctx.fill();
  ctx.restore();
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

// one upright strip from a to b with a line along its top, and the depth it is sorted by
function fence(ctx, P, [ax, ay], [bx, by], top, fill, edge) {
  return [P((ax + bx) / 2, (ay + by) / 2)[2], () => {
    ctx.fillStyle = fill;
    trace(ctx, P, [[ax, ay, 0], [bx, by, 0], [bx, by, top], [ax, ay, top]], true);
    ctx.fill();
    ctx.strokeStyle = edge;
    ctx.lineWidth = 2 * P(ax, ay, top)[3];
    trace(ctx, P, [[ax, ay, top], [bx, by, top]]);
    ctx.stroke();
  }];
}

// what stands up from the plate: the rim on its four sides and the marker strokes as low walls
function standing(ctx, P, maze, h) {
  const c = [[-h, h], [h, h], [h, -h], [-h, -h]];
  const rim = c.map((a, i) => fence(ctx, P, a, c[(i + 1) % 4], RIM_PX, SLAB[i], RIM));
  const wall = (a, b) => fence(ctx, P, a, b, WALL_PX, WALL_SIDE, WALL_TOP);
  const strokes = (maze && maze.walls) || [];
  return [...rim, ...strokes.flatMap(w => w.slice(1).map((b, i) => wall(w[i], b)))];
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

// the ball's place on the plate: crosshair and shadow
function mark(ctx, P, [x, y], h, r, k) {
  ctx.strokeStyle = CROSS;
  ctx.lineWidth = 1.5 * k;
  trace(ctx, P, [[x, -h, 1], [x, h, 1]]);
  ctx.stroke();
  trace(ctx, P, [[-h, y, 1], [h, y, 1]]);
  ctx.stroke();
  ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
  ring3(ctx, P, [x + 0.3 * r, y - 0.45 * r], 1.1 * r);
  ctx.fill();
}

// centre of the ball's tag: over the ball, under it when there is no room above, never on it.
// the ball is [x, y, radius] on the canvas, w and hgt the tag's size
function tagAt([sx, sy, rad], w, hgt, px) {
  const off = rad + hgt;
  return [Math.max(w / 2, Math.min(px - w / 2, sx)), sy - off < hgt / 2 ? sy + off : sy - off];
}

function tag(ctx, ball, text, px) {
  const size = Math.max((23 * px) / SCENE_PX, TAG_MIN_PX * devicePixelRatio);
  ctx.font = `${size}px 'VT323', monospace`;
  ctx.textAlign = "center";
  const w = ctx.measureText(text).width + size, hgt = 1.2 * size, [x, y] = tagAt(ball, w, hgt, px);
  ctx.fillStyle = "#0B0E1E";
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y - hgt / 2, w, hgt, hgt / 2);
  ctx.fill();
  ctx.fillStyle = "#FF7A6B";
  ctx.fillText(text, x, y + 0.26 * size);
}

function draw3d(ctx, cfg, maze, s, cam) {
  const px = fit(ctx.canvas), h = cfg.plate_inner_mm / 2, r = cfg.ball_d_mm / 2, u = PLATE_PX / h;
  const tilt = [s.plate_state.sent_x_deg, s.plate_state.sent_y_deg], P = projector(cam, tilt, h, px);
  const k = P(cam.cx, cam.cy)[3], at = [s.ball.x_mm, s.ball.y_mm], g = s.race && s.race.ghost_mm;
  const up = standing(ctx, P, maze, h);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, px, px);
  ctx.lineJoin = ctx.lineCap = "round";
  tray(ctx, P, h, px / SCENE_PX);
  if (maze) {
    flat(ctx, P, maze, r, s.race, k);
    up.push(...gateSigns(ctx, P, maze, s.race));
  }
  if (s.choices.mode === "balance") {
    ctx.strokeStyle = PATH;
    ctx.lineWidth = 3 * k;
    ring3(ctx, P, s.target.mm, 1.25 * r);
    ctx.stroke();
  }
  if (g) up.push(orb(ctx, P, g, r, u, ghostPaint(ctx)));
  if (s.ball.valid) mark(ctx, P, at, h, r, k);
  // far things first so near ones cover them
  up.sort((a, b) => b[0] - a[0]).forEach(([, paint]) => paint());
  if (!s.ball.valid) return;
  // the ball and its tag go on last, no wall or rim ever hides them
  const [sx, sy, , kb] = P(...at, r * u);
  orb(ctx, P, at, r, u, ballPaint(ctx))[1]();
  tag(ctx, [sx, sy, r * u * kb], `x ${signed(at[0], 1)}  y ${signed(at[1], 1)} mm`, px);
}
