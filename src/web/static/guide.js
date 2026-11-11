// one "do this next" line for the banner, first match wins; tone is "" | "warn" | "stop"
function guide(s, ctx) {
  s = s || {};
  const r = s.race && s.race.state ? s.race : {};
  const say = (text, tone = "") => ({ text, tone });
  if (!ctx.live) return say("connecting...", "warn");
  if (s.board_lost) return say("board lost, check the usb cable", "stop");
  if (s.camera_lost) return say("camera lost", "stop");
  if (s.killing || (!s.armed && s.stopped_by))
    return say(s.stopped_by ? "stopped: " + s.stopped_by : "stopped", "stop");
  if (ctx.watch) return say("");
  if (!s.armed && s.arm_blocked) return say(s.arm_blocked, "warn");
  if (s.armed && s.hand_calibrating) return say("hold still... locking neutral", "warn");
  if (s.armed && s.stale) return say("no input, plate levelled", "warn");
  if (ctx.lost) return say("HAND LOST", "stop");
  // the hud has these
  if (["countdown", "dnf"].includes(r.state)) return say("");
  if (r.state === "running") {
    return say(r.reached + 1 < ctx.gates ? `gate ${r.reached + 1} is next` : "head for the finish");
  }
  if (r.state === "finished") {
    if (typeof r.delta_s !== "number") return say("finished! type a name to save it");
    return say(r.delta_s < 0 ? "you beat the ghost!" : "the ghost wins, r to race again");
  }
  if (r.state === "calibrating") {
    return say((r.cal && r.cal.msg) || "calibrating... keep your hand off, place the ball in the middle", "warn");
  }
  if (r.state === "staging") return say("roll the ball into the start ring");
  if (!s.armed && !(s.ball && s.ball.valid)) return say("put the ball on the plate");
  if (!s.armed) return say("click arm to start");
  if (s.choices && s.choices.mode === "balance") {
    const lost = s.target.src === "marker" && !s.target.found;
    return lost ? say("no x on the board yet, draw one", "warn") : say("click the board or draw an x");
  }
  if (!ctx.hand) return say("arrows / wasd to tilt, r to race");
  if (!s.hand_seen) return say("hand flat under the webcam, palm down");
  return say("tilt your hand to roll the ball, r to race");
}
