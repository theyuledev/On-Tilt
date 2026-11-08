import copy
import select
import socket
from itertools import pairwise

import pytest

from src.control.pulse_map import center
from src.inputs.hand import HandInput, HandIntent, send_intent
from src.io.settings import load
from src.loop import Loop
from src.safety import limits

pytestmark = pytest.mark.sim

DT = 0.01
NEUTRAL = (0.0, 0.0)  # palm roll, pitch in degrees
# webcam above the hand looking down: right side down reads roll +,
# tipping the fingertips down (away from the player) reads pitch -
MOVES = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}


def palm(cfg, move=None, spans=1.0):
    if move is None:
        return NEUTRAL
    dx, dy = MOVES[move]
    d = spans * cfg["input"]["hand_tilt_span_deg"]
    return NEUTRAL[0] + dx * d, NEUTRAL[1] + dy * d


class Rig:
    """Hand process logic -> real udp socket -> HandInput -> Loop on the sim plate and ball."""

    def __init__(self, cfg, hand, sock):
        self.cfg, self.hand, self.sock = cfg, hand, sock
        self.palm = HandIntent(cfg["input"])
        self.loop = Loop(cfg, hand=hand)
        self.now = 1000.0
        self.recs = []

    def tick(self, at, send=True):
        self.now += DT
        if send:
            p = self.palm
            intent = p.update(at, self.now)
            flags = {"seen": p.seen, "calibrating": p.calibrating, "progress": p.progress}
            send_intent(self.sock, intent, self.hand.port, **flags)
            assert select.select([self.hand.sock], [], [], 1.0)[0], "packet never arrived"
        rec = self.loop.tick(self.now, DT)
        self.recs.append(rec)
        return rec

    def hold(self, s, at=None, send=True):
        for _ in range(round(s / DT)):
            rec = self.tick(at, send)
        return rec

    def lock(self):
        self.loop.arm()
        self.hold(self.cfg["input"]["neutral_s"] + 0.1, NEUTRAL)
        assert self.palm.neutral is not None

    @property
    def pulses(self):
        return [(r.servo_cmd.x_us, r.servo_cmd.y_us) for r in self.recs]


@pytest.fixture
def rig():
    cfg = copy.deepcopy(load())
    cfg["control"]["mode"] = "hand"
    hand = HandInput(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    yield Rig(cfg, hand, sock)
    sock.close()
    hand.close()


def level(rig):
    return tuple(round(v) for v in center(rig.cfg["rig"]))


def ramp_s(rig):
    r = rig.cfg["rig"]
    return (limits(r)[1] - center(r)[0]) / r["kill_ramp_us_s"]


def max_step(pulses):
    return max(max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in pairwise(pulses))


def assert_ramps_to_level(rig, pulses, after_s=0.0):
    rate = rig.cfg["rig"]["kill_ramp_us_s"]
    assert max_step(pulses) <= rate * DT + 1
    dist = max(abs(p - c) for p, c in zip(pulses[0], level(rig), strict=True))
    at = pulses.index(level(rig))
    assert at * DT <= after_s + dist / rate + 0.05
    assert set(pulses[at:]) == {level(rig)}


def tilt_right(rig):
    rig.lock()
    rig.hold(0.5, palm(rig.cfg, "right"))
    assert rig.pulses[-1][0] > level(rig)[0]
    return len(rig.recs) - 1


def test_holding_still_locks_neutral_with_the_plate_level(rig):
    rig.loop.arm()
    for _ in range(round(rig.cfg["input"]["neutral_s"] / DT)):
        rec = rig.tick(NEUTRAL)
        assert rig.loop.snapshot()["hand_calibrating"]
        assert rec.armed and not rec.intent.valid and rig.pulses[-1] == level(rig)
    rec = rig.hold(0.1, NEUTRAL)
    snap = rig.loop.snapshot()
    assert not snap["hand_calibrating"] and snap["hand_progress"] == 1.0
    assert rec.intent.valid


@pytest.mark.parametrize(
    ("move", "axis", "sign"), [("right", 0, 1), ("left", 0, -1), ("up", 1, 1), ("down", 1, -1)]
)
def test_each_move_rolls_the_ball_that_way(rig, move, axis, sign):
    rig.lock()
    rig.hold(0.5, palm(rig.cfg, move))
    pos = (rig.loop.ball.x_mm, rig.loop.ball.y_mm)
    assert sign * pos[axis] > 10
    assert pos[1 - axis] == pytest.approx(0, abs=0.5)


def test_dropout_ramps_the_plate_back_to_level_and_stays_armed(rig):
    start = tilt_right(rig)
    rec = rig.hold(ramp_s(rig) + 0.2, None)
    assert_ramps_to_level(rig, rig.pulses[start:])
    assert rec.armed and not rig.loop.snapshot()["hand_seen"]


def test_a_dead_hand_process_goes_stale_then_ramps_to_level(rig):
    start = tilt_right(rig)
    stale_s = rig.cfg["rig"]["stale_intent_s"]
    rig.hold(stale_s - 2 * DT, send=False)
    assert not rig.loop.snapshot()["stale"] and rig.pulses[-1] == rig.pulses[start]
    rec = rig.hold(ramp_s(rig) + 0.2, send=False)
    assert_ramps_to_level(rig, rig.pulses[start:], stale_s)
    assert rec.armed and not rig.loop.snapshot()["hand_seen"]


def test_a_palm_back_inside_the_keep_window_steers_without_a_new_hold(rig):
    tilt_right(rig)
    rig.hold(rig.cfg["input"]["neutral_keep_s"] / 2, None)
    assert rig.pulses[-1] == level(rig)
    rec = rig.tick(palm(rig.cfg, "up"))
    assert rec.intent.valid and not rig.loop.snapshot()["hand_calibrating"]
    # it eases in from level rather than snapping to the held tilt
    assert (rec.intent.x, rec.intent.y) == (0.0, 0.0)
    rec = rig.hold(rig.cfg["input"]["key_ramp_s"], palm(rig.cfg, "up"))
    assert rec.intent.y == pytest.approx(1.0)
    assert rig.pulses[-1][1] > level(rig)[1]


def test_the_plate_never_jumps_across_a_whole_hand_run(rig):
    tilt_right(rig)
    rig.hold(0.2, None)
    rig.hold(0.5, palm(rig.cfg, "left", spans=2.0))
    r = rig.cfg["rig"]
    assert max_step(rig.pulses) <= max(r["slew_us_s"], r["kill_ramp_us_s"]) * DT + 1
    locked = next(i for i, rec in enumerate(rig.recs) if rec.intent.valid)
    first = rig.pulses[locked]
    assert all(abs(p - c) <= 2 for p, c in zip(first, level(rig), strict=True))
    x_lo, x_hi, y_lo, y_hi = limits(r)
    assert all(x_lo <= x <= x_hi and y_lo <= y <= y_hi for x, y in rig.pulses)
    assert rig.pulses[-1][0] < level(rig)[0]
