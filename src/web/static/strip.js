// last few seconds of ball distance from the plate centre and sent tilt, each scaled to its own max
const STRIP_S = 5;
const ERR_COLOR = "#E69F00";
const TILT_COLOR = "#56B4E9";
const STRIP_LINE_PX = 2;
const STRIP_FONT_PX = 12;
const stripCtx = document.getElementById("strip").getContext("2d");
const stripBuf = [];

function stripLine(w, h, now, key, color) {
  const pad = STRIP_LINE_PX * devicePixelRatio;
  const top = Math.max(0, ...stripBuf.map(p => p[key] ?? 0)) || 1;
  stripCtx.lineWidth = pad;
  stripCtx.strokeStyle = color;
  stripCtx.beginPath();
  let down = false;
  for (const p of stripBuf) {
    if (p[key] === null) { down = false; continue; }
    const x = w * (1 - (now - p.t) / STRIP_S), y = h - pad - (h - 2 * pad) * p[key] / top;
    if (down) stripCtx.lineTo(x, y); else stripCtx.moveTo(x, y);
    down = true;
  }
  stripCtx.stroke();
}

function strip(s) {
  const b = s.ball, p = s.plate_state;
  const err = b.valid ? Math.hypot(b.x_mm, b.y_mm) : null;
  const tilt = Math.hypot(p.sent_x_deg, p.sent_y_deg);
  // stored as x = err mm (null leaves a gap), y = tilt deg
  keep(stripBuf, [err, tilt], s.t_s, STRIP_S);
  const c = stripCtx.canvas, w = fit(c), h = Math.round(c.clientHeight * devicePixelRatio);
  if (c.height !== h) c.height = h;
  stripCtx.clearRect(0, 0, w, h);
  stripLine(w, h, s.t_s, "x", ERR_COLOR);
  stripLine(w, h, s.t_s, "y", TILT_COLOR);
  const fs = STRIP_FONT_PX * devicePixelRatio;
  stripCtx.font = `${fs}px system-ui, sans-serif`;
  stripCtx.textBaseline = "top";
  const errText = err === null ? "err --" : `err ${Math.round(err)} mm`;
  stripCtx.fillStyle = ERR_COLOR;
  stripCtx.fillText(errText, fs / 2, fs / 3);
  stripCtx.fillStyle = TILT_COLOR;
  stripCtx.fillText(`tilt ${tilt.toFixed(1)}°`, fs * 1.5 + stripCtx.measureText(errText).width, fs / 3);
}
