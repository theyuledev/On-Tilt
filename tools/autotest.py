import math
import sys

from src.app import Loop, check_exposure
from src.contracts import BallState
from src.io.camera import CameraSource, open_camera
from src.io.serial_link import SerialActuator, open_board
from src.io.settings import load
from src.sim.fit import RELEASE_MM, hold_verdict
from src.vision.pipeline import Pipeline
from tools.autotest_board import Feed, Result, Rig, board_checks, to_level

HOLD_S = 35.0
FIND_S = 5.0


def holding(rig: Rig, loop: Loop) -> str | None:
    # where assist stops short of the rim
    margin = rig.cfg["control"]["assist_margin_mm"]
    rim = (rig.r["plate_inner_mm"] - rig.r["ball_d_mm"]) / 2 - margin
    for _ in range(rig.ticks(HOLD_S)):
        rec = rig.tick(loop, 0.0)
        if not rec.armed:
            return "board lost" if loop.board_lost else "board disarmed"
        if not rec.ball.valid:
            return "ball lost"
        if math.hypot(rec.ball.x_mm, rec.ball.y_mm) > rim:
            return f"ball near the rim at ({rec.ball.x_mm:.0f}, {rec.ball.y_mm:.0f}) mm"
    return None


def hold(rig: Rig) -> list[Result]:
    loop = rig.hold_loop(rig)
    loop.keys, loop.mode = Feed(), "balance"
    for _ in range(rig.ticks(FIND_S)):
        if rig.tick(loop, 0.0).ball.valid:
            break
    trace = []
    loop.bus.subscribe("tick", trace.append)
    loop.arm()
    try:
        why = holding(rig, loop) if loop.safety.armed else f"arm refused: {loop.arm_blocked}"
    finally:
        to_level(rig, loop)
    if why:
        return [("balance hold", why)]
    rows = ((r.t_s, r.ball.x_mm, r.ball.y_mm) for r in trace[: rig.ticks(HOLD_S)])
    t, x, y = zip(*rows, strict=True)
    verdict = hold_verdict(t, x, y).items()
    shown = ((k, ok, "n/a" if v is None else f"{v:.2f}") for k, (ok, v) in verdict)
    return [(f"hold {k} {v}", None if ok else v) for k, ok, v in shown]


def show(results: list[Result]) -> bool:
    for name, why in results:
        print(f"ok  {name}" if why is None else f"FAIL {name}: {why}")
    return all(why is None for _, why in results)


def sim_hold(rig: Rig) -> Loop:
    loop = Loop(rig.cfg)
    loop.ball = BallState(RELEASE_MM, 0.0, 0.0, 0.0, valid=True, t_s=rig.now())
    return loop


def real_hold(rig: Rig) -> Loop:
    input(f"set the ball down about {RELEASE_MM:g} mm out on +x and let go, then enter ")
    frames = CameraSource(open_camera(rig.r), process=Pipeline(rig.cfg).step)
    actuator = SerialActuator(rig.link, rig.r)
    return Loop(rig.cfg, frames=frames, actuator=actuator, exposure=check_exposure(rig.r))


class Clock:
    t = 1000.0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def sim_rig(cfg: dict, board=None) -> Rig:
    # the timed fake board lives with the tests, real mode never needs it
    from tests.contract.timed_board import TimedBoard

    clock = Clock()
    return Rig(cfg, (board or TimedBoard)(clock.now), sim_hold, clock.now, clock.sleep)


def main() -> None:
    cfg = load()
    sim = "--sim" in sys.argv[1:]
    rig = sim_rig(cfg) if sim else Rig(cfg, open_board(cfg["rig"]).port, real_hold)
    if not show(board_checks(rig) + hold(rig)):
        sys.exit(1)


if __name__ == "__main__":
    main()
