// last few seconds of ball distance from the plate centre (full height = plate half width)
// and sent tilt (scaled to its own max)
const STRIP_S = 5;
const ERR_COLOR = "#E69F00";
const TILT_COLOR = "#56B4E9";
const STRIP_LINE_PX = 2;
const stripCtx = document.getElementById("strip").getContext("2d");
const stripBuf = [];

function stripLine(w, h, now, key, top, color) {
  const pad = STRIP_LINE_PX * devicePixelRatio;
  stripCtx.lineWidth = pad;
  stripCtx.strokeStyle = color;
  stripCtx.beginPath();
  let down = false;
  for (const p of stripBuf) {
    if (p[key] === null) { down = false; continue; }
    const x = w * (1 - (now - p.t) / STRIP_S), y = h - pad - (h - 2 * pad) * Math.min(p[key] / top, 1);
    if (down) stripCtx.lineTo(x, y); else stripCtx.moveTo(x, y);
    down = true;
  }
  stripCtx.stroke();
}

function strip(cfg, s) {
  const b = s.ball, p = s.plate_state;
  const err = b.valid ? Math.hypot(b.x_mm, b.y_mm) : null;
  const tilt = Math.hypot(p.sent_x_deg, p.sent_y_deg);
  // stored as x = err mm (null leaves a gap), y = tilt deg
  keep(stripBuf, [err, tilt], s.t_s, STRIP_S);
  const c = stripCtx.canvas, w = fit(c), h = Math.round(c.clientHeight * devicePixelRatio);
  if (c.height !== h) c.height = h;
  stripCtx.clearRect(0, 0, w, h);
  stripLine(w, h, s.t_s, "x", cfg.plate_inner_mm / 2, ERR_COLOR);
  stripLine(w, h, s.t_s, "y", Math.max(...stripBuf.map(q => q.y)) || 1, TILT_COLOR);
  document.getElementById("striperr").textContent = err === null ? "ERR --" : `ERR ${Math.round(err)} mm`;
  document.getElementById("striptilt").textContent = `TILT ${tilt.toFixed(1)}°`;
}
