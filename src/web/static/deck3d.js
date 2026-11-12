// what lies on the 3D plate under the ball: the path, the marker strokes, a course's start and
// gate rings and the numbers over them (uses plate3d.js trace, race.js colours, gates.js GATE /
// GATE_LIT / NUM_FONT / startOpen / startInk)
const LABEL_PX = 56;  // how high a gate's number hangs over the plate
const RING = Array.from({ length: 25 }, (_, i) => [Math.cos((i * Math.PI) / 12), Math.sin((i * Math.PI) / 12)]);

function ring3(ctx, P, [x, y], r, z = 1) {
  trace(ctx, P, RING.map(([c, s]) => [x + r * c, y + r * s, z]), true);
}

// the path and the marker strokes, flat on the plate
function flat(ctx, P, path, maze, k) {
  ctx.strokeStyle = PATH;
  ctx.lineWidth = 2 * k;
  ctx.setLineDash([6 * k, 6 * k]);
  trace(ctx, P, (path || []).map(([x, y]) => [x, y, 1]));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = MARKER;
  ctx.lineWidth = 7 * k;
  ((maze && maze.walls) || []).forEach(w => { trace(ctx, P, w.map(([x, y]) => [x, y, 1])); ctx.stroke(); });
}

// a course's start ring and gate rings, each at the radius that takes it
function rings(ctx, P, maze, race, takeR, k) {
  const reached = race && race.state ? race.reached : 0;
  ctx.lineWidth = 2 * k;
  maze.checkpoints.forEach((i, n) => {
    ctx.strokeStyle = n < reached ? GATE_LIT : n === reached ? PATH : GATE;
    ring3(ctx, P, maze.points[i], takeR);
    ctx.stroke();
  });
  // after the gates: a lap's last gate sits on the start and would cover its ring
  if (startOpen(race)) {
    ctx.strokeStyle = startInk(race);
    ctx.setLineDash([6 * k, 6 * k]);
    ring3(ctx, P, maze.start, takeR);
    ctx.stroke();
    ctx.setLineDash([]);
  }
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
