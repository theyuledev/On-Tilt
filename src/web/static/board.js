// the edge guard on the plate picture: a dashed line inside the rim, solid on the side it is holding
const GUARD = "#B36B00";
const GUARD_INSET_MM = 13;

// the one place that knows the snapshot's guard field
function guardOf(s) {
  const g = s.guard;
  if (!g || !Array.isArray(g.w) || !Array.isArray(g.side)) return null;
  return { w: g.w.map(v => Number(v) || 0), side: g.side.map(v => Math.sign(v)) };
}

function glowAlpha(w) {
  return w < 0.05 ? 0 : Math.max(0.15, Math.min(0.6, w));
}

// h is the plate half width in mm, ctx is already in plate mm
function guardLine(ctx, s, h) {
  const g = guardOf(s), e = h - GUARD_INSET_MM;
  if (!g) return;
  ctx.strokeStyle = GUARD;
  ctx.lineWidth = 0.9;
  ctx.setLineDash([2, 3]);
  ctx.strokeRect(-e, -e, 2 * e, 2 * e);
  ctx.setLineDash([]);
  ctx.lineWidth = 3;
  [0, 1].forEach(axis => {
    const at = g.side[axis] * e, a = glowAlpha(g.w[axis]);
    if (!at || !a) return;
    ctx.globalAlpha = a;
    polyline(ctx, axis ? [[-e, at], [e, at]] : [[at, -e], [at, e]]);
  });
  ctx.globalAlpha = 1;
}
