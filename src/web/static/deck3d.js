// what lies on the 3D plate under the ball, drawn with the flat view's own code: through flatPen
// for lines and rings of any size, through decal for the small things with text or letters in
// them (uses plate3d.js trace, cam.js nearest, race.js polyline / walls / tail / trail buffers and
// colours, gates.js gates / gate / ends / letter / zone, board.js guardLine, target.js reticle)
const RIBBON_MM = 4;  // width of the path still to drive
const DONE_MM = 2;  // width of the part already driven
const RING = Array.from({ length: 25 }, (_, i) => [Math.cos((i * Math.PI) / 12), Math.sin((i * Math.PI) / 12)]);

function ring3(ctx, P, [x, y], r, z = 1) {
  trace(ctx, P, RING.map(([c, s]) => [x + r * c, y + r * s, z]), true);
}

// a pen in plate mm: what is drawn with it lands on the tilted plate. mm is canvas px per plate mm.
// everything it does not change goes to the canvas itself, called as the canvas
function flatPen(ctx, P, mm) {
  const own = {
    moveTo: (x, y) => ctx.moveTo(...P(x, y, 1).slice(0, 2)),
    lineTo: (x, y) => ctx.lineTo(...P(x, y, 1).slice(0, 2)),
    arc: (x, y, r) => ring3(ctx, P, [x, y], r),
    strokeRect: (x, y, w, h) => {
      trace(ctx, P, [[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]], true);
      ctx.stroke();
    },
    setLineDash: dash => ctx.setLineDash(dash.map(v => v * mm)),
  };
  return new Proxy(ctx, {
    get: (t, k) => own[k] || (typeof t[k] === "function" ? t[k].bind(t) : t[k]),
    set: (t, k, v) => { t[k] = k === "lineWidth" ? v * mm : v; return true; },
  });
}

// paints something small that lies on the plate at [x, y]: inside paint the canvas is in plate mm
// with +y up, as in the flat view. straight lines stay exact, the size is right at [x, y]
function decal(ctx, P, [x, y], paint) {
  const [ox, oy] = P(x, y, 1), [ax, ay] = P(x + 1, y, 1), [bx, by] = P(x, y + 1, 1);
  const a = ax - ox, b = ay - oy, c = bx - ox, d = by - oy;
  ctx.save();
  ctx.setTransform(a, b, c, d, ox - a * x - c * y, oy - b * x - d * y);
  paint();
  ctx.restore();
}

// the path cut at the ball: [driven, still to go]. in a race the ball is looked for only between
// the last gate taken and the next, so a lane beside an earlier one cannot take it
function pathSplit(path, maze, race, ball) {
  const st = race && race.state;
  if (st === "finished") return [path, []];
  if (st !== "running" || !ball.valid) return [[], path];
  const cps = maze.checkpoints, from = race.reached ? cps[race.reached - 1] : 0;
  const hit = nearest(path, [ball.x_mm, ball.y_mm], from, cps[race.reached] ?? path.length - 1);
  return hit ? [[...path.slice(0, hit.i + 1), hit.at], [hit.at, ...path.slice(hit.i + 1)]] : [[], path];
}

function ribbon(pen, [done, togo]) {
  pen.strokeStyle = PATH;
  pen.lineWidth = RIBBON_MM;
  polyline(pen, togo);
  pen.strokeStyle = GATE_LIT;
  pen.lineWidth = DONE_MM;
  polyline(pen, done);
}

// everything flat, in the flat view's order. path is what shownPath gave, mm canvas px per plate mm
function deck(ctx, P, cfg, maze, s, path, mm) {
  const pen = flatPen(ctx, P, mm), h = cfg.plate_inner_mm / 2, r = cfg.ball_d_mm / 2;
  const course = !!maze && path === maze.points;
  guardLine(pen, s, h);
  if (maze && maze.walls) walls(pen, maze);
  if (course) ends(ctx, maze, h, r, (_, at, strokes) => decal(ctx, P, at, () => letter(ctx, at, strokes)));
  if (path) ribbon(pen, pathSplit(path, maze, s.race, s.ball));
  if (course) {
    const badge = (_, at, ...rest) => decal(ctx, P, at, () => gate(ctx, at, ...rest));
    gates(pen, maze, r, s.race, cfg.checkpoint_r_mm, badge);
  }
  if (s.race && s.race.ghost_mm) tail(pen, ghostTrail, s.t_s, TRAIL_S, 1.6, GHOST_EDGE, 0.6);
  if (s.choices.mode === "balance") reticle(pen, s.target, r);
  tail(pen, trail, s.t_s, TRAIL_S, r / 2, BALL, 0.25);
}

// the next gate's ring once more, over whatever stands in front of it
function nextGate(ctx, P, maze, race, takeR, mm) {
  const i = race && race.state === "running" ? maze.checkpoints[race.reached] : undefined;
  if (i !== undefined) zone(flatPen(ctx, P, mm), maze.points[i], takeR, PATH);
}
