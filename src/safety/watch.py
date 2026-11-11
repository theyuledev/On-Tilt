"""When an armed plate must stop on its own: it swings hard, sits at its limit, the ball does not
answer it, or the camera is gone. Pure rules over the last few seconds of ticks."""

from src.control.pulse_map import center
from src.modes import CLOSED_LOOP

HZ = 20
STOP = (
    "plate is swinging hard",
    "plate held at its limit",
    "ball is not answering the plate",
    "camera lost",
)


def row(s: dict, mid: tuple[float, float]) -> dict:
    """The few numbers the checks need from one snapshot or run log record."""
    return {
        "t": s["t_s"],
        "armed": bool(s["armed"]),
        # a player may hold full tilt or push into a wall, a mode steering on the ball may not
        "auto": s["mode"] in CLOSED_LOOP,
        "seen": bool(s["ball"]["valid"]),
        "pos": (s["ball"]["x_mm"], s["ball"]["y_mm"]),
        "vel": (s["ball"]["vx_mm_s"], s["ball"]["vy_mm_s"]),
        "us": (s["servo_cmd"]["x_us"] - mid[0], s["servo_cmd"]["y_us"] - mid[1]),
        "deg": (s["plate_state"]["sent_x_deg"], s["plate_state"]["sent_y_deg"]),
        "board_lost": bool(s.get("board_lost")),
        "camera_lost": bool(s.get("camera_lost")),
    }


def swings(us: list[float], big: float) -> int:
    """Turns of the pulse where it had run at least big us since the turn before."""
    n, ext, heading = 0, us[0], 0
    for u in us[1:]:
        if heading == 0:
            if abs(u - ext) >= big:
                heading, ext = (1 if u > ext else -1), u
        elif (u - ext) * heading > 0:
            ext = u
        elif (ext - u) * heading >= big:
            n, heading, ext = n + 1, -heading, u
    return n


def held(rows: list[dict], want, span_s: float) -> bool:
    """want held for every row of the last span_s, and the rows reach that far back."""
    now = rows[-1]["t"]
    tail = [r for r in rows if now - r["t"] <= span_s]
    return rows[0]["t"] <= now - span_s and all(want(r) for r in tail)


def problems(rows: list[dict], rig: dict, c: dict) -> list[str]:
    """What is wrong right now, from the last few seconds of rows (oldest first)."""
    last = rows[-1]
    out = [
        name
        for name, key in (("board link lost", "board_lost"), ("camera lost", "camera_lost"))
        if last[key]
    ]
    if not last["armed"]:
        return out
    if held(rows, lambda r: r["armed"] and not r["seen"], c["watch_blind_s"]):
        out.append("ball not seen")
    window = [r for r in rows if last["t"] - r["t"] <= c["watch_window_s"] and r["armed"]]
    for i, limit in enumerate((rig["limit_x_us"], rig["limit_y_us"])):
        if swings([r["us"][i] for r in window], c["watch_swing_us"]) >= c["watch_swings"]:
            out.append("plate is swinging hard")
        if held(
            rows,
            lambda r, i=i, limit=limit: r["auto"] and r["armed"] and abs(r["us"][i]) >= limit - 5,
            c["watch_pinned_s"],
        ):
            out.append("plate held at its limit")
        if deaf(rows, i, rig, c):
            out.append("ball is not answering the plate")
    return list(dict.fromkeys(out))


def deaf(rows: list[dict], i: int, rig: dict, c: dict) -> bool:
    """A steady tilt one way for watch_deaf_s that the ball's speed did not follow: a servo
    that is not moving, a horn off its link, a wire holding the plate."""
    span, tilt = c["watch_deaf_s"], c["watch_deaf_deg"]
    side = 1 if rows[-1]["deg"][i] > 0 else -1
    inner = rig["plate_inner_mm"] / 2 - 2 * rig["ball_d_mm"]

    def pushed(r):
        steady = r["auto"] and r["armed"] and r["seen"] and r["deg"][i] * side >= tilt
        return steady and abs(r["pos"][i]) < inner

    if not held(rows, pushed, span):
        return False
    tail = [r for r in rows if rows[-1]["t"] - r["t"] <= span]
    gained = (tail[-1]["vel"][i] - tail[0]["vel"][i]) * side
    return gained < c["watch_deaf_frac"] * c["sim_g_mm_s2"] * 5 / 7 * 0.0175 * tilt * span


class Watch:
    """The last few seconds of ticks, and why the plate was stopped if it was."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.reason: str | None = None

    def see(self, rec, camera_lost: bool, cfg: dict) -> bool:
        """Takes one tick's record. True when the plate must stop now."""
        if self.rows and rec.t_s - self.rows[-1]["t"] < 1 / HZ:
            return False
        s = rec.to_dict() | {"camera_lost": camera_lost}
        self.rows = [*self.rows, row(s, center(cfg["rig"]))][-8 * HZ :]
        if not rec.armed:
            return False
        bad = [p for p in problems(self.rows, cfg["rig"], cfg["control"]) if p in STOP]
        if bad:
            self.reason = bad[0]
        return bool(bad)
