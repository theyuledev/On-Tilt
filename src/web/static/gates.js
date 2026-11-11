// the checkpoint rings on the board picture (uses race.js circle and its colours)
const GATE = "#4A5380";
const GATE_LIT = "#00795A";
const NUM_FONT = "'Press Start 2P', monospace";
const GATE_R = 11 / 8;  // gate ring in ball radii
const GATE_PULSE_MS = 600;
const GATE_FLASH_MS = 120;
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
  ctx.fillStyle = GATE;
  circle(ctx, maze.start, 4);
  ctx.fill();
}
