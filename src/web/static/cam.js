// camera director for the follow and 3D views: which shot runs and where it looks, eased so a
// noisy, jumpy or briefly lost ball never shakes or snaps the picture. plate mm, +y up, no page state
const HOLD_S = 1.0;  // a shot stays up at least this long
const WIDE_S = 1.2;  // a race opens on the wide shot
const LOST_S = 1.0;  // a ball gone this long sends the camera back to wide
const JUMP_MM = 60;  // no ball moves this far between two frames, so it is the tracker jumping
const JUMP_S = 0.15;  // unless it stays there this long, then the ball really is there now
const NEAR_MM = 40;  // the gate cam cuts in this far before the ball takes the next gate
const LEAD_MM = 20;  // chase looks this far ahead of the ball
const EYE_RATE = 6;  // 1/s, how fast the camera's idea of the ball and of the heading follow
const CAM_RATE = 4;  // 1/s, how fast the camera eases into a shot
const DT_MAX_S = 0.05;
const SHOT_NAMES = { tower: "CAM 1 TOWER", wide: "CAM 2 WIDE", chase: "CAM 3 CHASE", gate: "CAM 4 GATE" };

function camStart() {
  return { t: null, shot: "wide", shotAt: -Infinity, lostS: 0, odd: null, oddS: 0, last: null,
           eye: null, dir: [0, 1], filmed: null, cam: { pitch: 56, yaw: 0, zoom: 1.2, cx: 0, cy: 0 } };
}

// the point of the path nearest to p, looked for on the segments between points from and to:
// { i: the segment, at: the point, dir: the segment's unit direction }, null when it has no length
function nearest(path, p, from = 0, to = path.length - 1) {
  let best = Infinity, hit = null;
  for (let i = from; i < to; i++) {
    const [ax, ay] = path[i], dx = path[i + 1][0] - ax, dy = path[i + 1][1] - ay;
    const len = Math.hypot(dx, dy);
    if (!len) continue;
    const u = Math.max(0, Math.min(len, ((p[0] - ax) * dx + (p[1] - ay) * dy) / len));
    const at = [ax + (dx * u) / len, ay + (dy * u) / len], d = Math.hypot(at[0] - p[0], at[1] - p[1]);
    if (d < best) { best = d; hit = { i, at, dir: [dx / len, dy / len] }; }
  }
  return hit;
}

// unit direction of the path segment nearest to p, null when the path has no length
function pathDir(path, p) {
  const hit = nearest(path, p);
  return hit && hit.dir;
}

const within = (gate, eye, mm) => !!gate && Math.hypot(gate[0] - eye[0], gate[1] - eye[1]) < mm;
// the next gate counts as near from NEAR_MM outside the ring that takes it
const closing = (inp, eye) => within(inp.gate, eye, (inp.gateR || 0) + NEAR_MM);

// inp.phase is "idle", "run" or "done"; with no path there is no heading, so it stays wide.
// filmed is the gate a running gate shot is on, it stays on it until the ball is clear of it
function wantShot(inp, eye, lostS, filmed) {
  if (inp.phase === "done") return "tower";
  if (inp.phase !== "run" || !inp.path || inp.runS < WIDE_S || lostS > LOST_S) return "wide";
  return closing(inp, eye) || within(filmed, eye, NEAR_MM) ? "gate" : "chase";
}

// yaw turns the plate so the heading points up the screen
function shotGoal(shot, t, eye, dir, gate) {
  const heading = 90 - (Math.atan2(dir[1], dir[0]) * 180) / Math.PI;
  if (shot === "tower") return { pitch: 0, yaw: 0, zoom: 1.15, cx: 0, cy: 0 };
  if (shot === "wide") return { pitch: 56, yaw: 12 * Math.sin(t * 0.5), zoom: 1.2, cx: 0, cy: 0 };
  if (shot === "chase") {
    return { pitch: 64, yaw: heading, zoom: 2, cx: eye[0] + dir[0] * LEAD_MM, cy: eye[1] + dir[1] * LEAD_MM };
  }
  const to = gate || eye;
  return { pitch: 72, yaw: heading + 32, zoom: 2.4, cx: (eye[0] + to[0]) / 2, cy: (eye[1] + to[1]) / 2 };
}

// inp: { t, ball: { x, y, valid }, path: [[x, y], ...] or null, gate: next gate [x, y] or null,
//        gateR: how close takes a gate, phase, runS: seconds since the race clock started }
function camStep(st, inp) {
  if (st.t !== null && inp.t < st.t) st = camStart();  // the app restarted and its clock with it
  const dt = st.t === null ? 0 : Math.min(DT_MAX_S, inp.t - st.t);
  const ease = rate => 1 - Math.exp(-dt * rate);
  const toward = (from, to, rate) => from.map((v, i) => v + (to[i] - v) * ease(rate));
  const b = inp.ball, at = [b.x, b.y];
  // a far off ball is only believed once it has stayed in that one new place for a while
  const far = b.valid && !within(st.last || at, at, JUMP_MM), stays = far && within(st.odd, at, JUMP_MM);
  const odd = !far ? null : stays ? st.odd : at, oddS = stays ? st.oddS + dt : 0;
  const ok = b.valid && (!far || oddS > JUMP_S);
  // a lost ball leaves the eye where it was, the first sighting puts it on the ball at once
  const seen = !ok ? st.eye : st.eye ? toward(st.eye, at, EYE_RATE) : at;
  const eye = seen || [0, 0], lostS = ok ? 0 : st.lostS + dt;
  const dir = toward(st.dir, (inp.path && pathDir(inp.path, eye)) || st.dir, EYE_RATE);
  const want = wantShot(inp, eye, lostS, st.filmed);
  // the finish and the fall back to wide cut at once, anything else waits out the hold
  const cut = want !== st.shot && (want === "tower" || want === "wide" || inp.t - st.shotAt >= HOLD_S);
  const shot = cut ? want : st.shot;
  const filmed = shot !== "gate" ? null : closing(inp, eye) || !st.filmed ? inp.gate : st.filmed;
  const goal = shotGoal(shot, inp.t, eye, dir, filmed), cam = {};
  goal.yaw += 360 * Math.round((st.cam.yaw - goal.yaw) / 360);  // the short way round
  for (const k of Object.keys(goal)) cam[k] = st.cam[k] + (goal[k] - st.cam[k]) * ease(CAM_RATE);
  const last = ok ? at : st.last;
  const shotAt = cut ? inp.t : st.shotAt;
  return { t: inp.t, shot, shotAt, lostS, odd, oddS, last, eye: seen, dir, filmed, cam };
}
