// which picture of the board this tab shows: plain top-down, the follow cam or the 3D plate
// it is only how the tab draws, nothing is sent. V steps through them, T is top-down at once
// (uses cam.js, follow.js, plate3d.js, race.js draw and app.js $ / typing)
const VIEWS = ["top", "follow", "3d"];
const TOP = { x: 0, y: 0, zoom: 1, tall: false };
const miniCtx = document.getElementById("mini").getContext("2d");
let view = "top", camSt = camStart();

function nextView() {
  view = VIEWS[(VIEWS.indexOf(view) + 1) % VIEWS.length];
}

addEventListener("keydown", e => {
  if (typing(e) || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === "KeyV") nextView();
  if (e.code === "KeyT") view = "top";
});
document.getElementById("viewtag").onclick = e => { nextView(); e.currentTarget.blur(); };

// what the camera director needs from a snapshot
function camInput(s, maze, cfg) {
  const r = s.race && s.race.state ? s.race : null, b = s.ball;
  const run = !!r && r.state === "running", next = maze && r ? maze.checkpoints[r.reached] : undefined;
  return {
    t: s.t_s,
    ball: { x: b.x_mm, y: b.y_mm, valid: b.valid },
    path: maze && maze.points.length > 1 ? maze.points : null,
    gate: next === undefined ? null : maze.points[next],
    gateR: cfg.checkpoint_r_mm,
    phase: r && r.state === "finished" ? "done" : run ? "run" : "idle",
    runS: run ? r.elapsed_s : 0,
  };
}

function drawView(ctx, cfg, maze, s) {
  camSt = camStep(camSt, camInput(s, maze, cfg));
  const follow = view === "follow";
  const look = follow ? followLook(camSt.eye || [0, 0], cfg.ball_d_mm / 2) : TOP;
  ctx.canvas.className = follow ? "follow" : "";
  $("minimap").hidden = !follow;
  $("viewtag").textContent = follow ? "FOLLOW CAM 3X" : view === "3d" ? SHOT_NAMES[camSt.shot] : "TOP VIEW";
  if (view === "3d") return draw3d(ctx, cfg, maze, s, camSt.cam);
  draw(ctx, cfg, maze, s, look);
  if (follow) minimap(miniCtx, cfg, maze, s, look);
}
