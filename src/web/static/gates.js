// the checkpoint rings and the S / E letters on the board picture
// (uses race.js circle and its colours)
const GATE = "#4A5380";
const GATE_LIT = "#00795A";
const NUM_FONT = "'Press Start 2P', monospace";
const GATE_R = 11 / 8;  // gate ring in ball radii
const GATE_PULSE_MS = 600;
const GATE_FLASH_MS = 120;
const LETTER_MM = [14.5, 11.5];  // letter centre out from the path end, for a ball of 8 mm radius
// marker strokes round the letter centre, y down
const S_STROKES = "M3 -5C0 -8 -5 -6 -3 -2.5C-1 1 3.5 -0.5 4 3C4.5 6.5 -1 8 -4 5";
const E_STROKES = "M3.5 -6H-3.5V6H3.5M-3.5 0H2.5";
let gatesHit = 0, gateHitAt = -Infinity;

// upright text centred on [x, y] mm
function label(ctx, [x, y], size, text, color, font = NUM_FONT) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(1, -1);
  ctx.font = `${size}px ${font}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function gate(ctx, at, r, n, hit, next, last) {
  const edge = hit ? GATE_LIT : next ? PATH : GATE;
  circle(ctx, at, r);
  ctx.fillStyle = hit ? GATE_LIT : BOARD;
  ctx.fill();
  ctx.strokeStyle = edge;
  ctx.lineWidth = next ? 2.6 : 1.6;
  if (!hit) ctx.stroke();
  if (!last) return label(ctx, at, (r * 8) / 11, String(n + 1), hit ? "#FFFFFF" : edge);
  ctx.lineWidth = hit ? 2 : 1.6;
  circle(ctx, at, r + 3.5);
  ctx.stroke();
}

function gates(ctx, maze, ballR, race) {
  const r = ballR * GATE_R, finish = maze.checkpoints.length - 1, now = performance.now();
  const reached = race && race.state ? race.reached : 0;
  const live = race && (race.state === "countdown" || race.state === "running");
  const pulse = 0.6 + 0.4 * Math.cos((2 * Math.PI * now) / GATE_PULSE_MS);
  if (reached > gatesHit) gateHitAt = now;
  gatesHit = reached;
  maze.checkpoints.forEach((i, n) => {
    const next = live && n === reached;
    ctx.globalAlpha = next ? pulse : 1;
    gate(ctx, maze.points[i], r, n, n < reached, next, n === finish);
  });
  ctx.globalAlpha = 1;
  const since = now - gateHitAt;
  if (reached && since < GATE_FLASH_MS) {
    ctx.strokeStyle = GATE_LIT;
    ctx.lineWidth = 3;
    circle(ctx, maze.points[maze.checkpoints[reached - 1]], r * (1 + since / GATE_FLASH_MS));
    ctx.stroke();
  }
}

// centre of the S or the E: beside its end of the path, out towards the nearer corner, on the plate
function letterAt([x, y], h, r) {
  const put = (v, d, edge) => Math.max(-edge, Math.min(edge, v + ((v < 0 ? -d : d) * r) / 8));
  return [put(x, LETTER_MM[0], h - 8), put(y, LETTER_MM[1], h - 11)];
}

function letter(ctx, [x, y], strokes) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(1, -1);
  ctx.lineJoin = ctx.lineCap = "round";
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = MARKER;
  ctx.stroke(new Path2D(strokes));
  ctx.restore();
}

// a lap that ends where it starts has only the S
function ends(ctx, maze, h, r) {
  const a = maze.start, b = maze.points[maze.points.length - 1];
  letter(ctx, letterAt(a, h, r), S_STROKES);
  if (b[0] !== a[0] || b[1] !== a[1]) letter(ctx, letterAt(b, h, r), E_STROKES);
}
