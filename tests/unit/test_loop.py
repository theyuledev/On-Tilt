import json
import math

import numpy as np
import pytest

from src.contracts import BallState, Intent, PlateCommand
from src.io.camera import ClipSource
from src.io.ports import Frame
from src.io.settings import ROOT, load
from src.loop import LEVEL, Loop
from src.modes import CLOSED_LOOP, MODES, make_mode
from src.vision.pipeline import Observation, Pipeline
from src.vision.tracker import BallPx

pytestmark = pytest.mark.unit

MISMATCH = "exposure 120 (100 us steps), wanted 30"
UNVERIFIED = "exposure unverified: uvc-util not on PATH"
CLIP = ROOT / "data" / "clips" / "squash_ball_hand_rolled_1944.avi"
START, END = 5130, 5370


class Cam:
    def __init__(self):
        self.queue, self.obs = [], None

    def read(self):
        return self.queue.pop(0) if self.queue else None

    def latest(self):
        return self.obs

    def close(self):
        pass

    def see(self, x_mm, t_s, valid=True):
        self.queue.append(Frame(np.zeros((1, 1)), t_s))
        ball = BallState(x_mm, 0.0, 0.0, 0.0, valid, t_s)
        px = BallPx(0.0, 0.0, 10.0, valid, t_s)
        self.obs = Observation(ball, px, math.nan, math.nan, False, t_s, 1.0)


class Hand:
    seen = calibrating = False
    progress = 0.0

    def __init__(self, x=0.0):
        self.x, self.t = x, 0.0

    def latest(self):
        return Intent(self.x, 0.0, True, self.t)

    def recentre(self):
        pass


class Spy:
    def __init__(self, valid=True):
        self.calls, self.valid = [], valid

    def update(self, t, x, y):
        self.calls.append(("update", t, x, y))

    def miss(self, t):
        self.calls.append(("miss", t))

    def command(self, t, cmd):
        self.calls.append(("command", t, cmd))

    def predict(self, now):
        return BallState(1.0, 2.0, 3.0, 4.0, self.valid, now)

    def seen(self):
        return [c for c in self.calls if c[0] != "command"]


def broken(call):
    spy = Spy()

    def boom(*a):
        raise ValueError("singular")

    setattr(spy, call, boom)
    return spy


class Board:
    armed = True

    def send(self, cmd, now):
        return LEVEL


def cam_loop(mode="balance"):
    cam = Cam()
    loop = Loop(load(), frames=cam)
    loop.mode, loop.est = mode, Spy()
    return loop, cam


def test_each_pipeline_result_reaches_the_estimator_once():
    loop, cam = cam_loop()
    cam.see(50.0, 1.0)
    for t in (1.0, 1.01, 1.02):
        loop.tick(t, 0.01)
    cam.see(0.0, 1.03, valid=False)
    for t in (1.03, 1.04):
        loop.tick(t, 0.01)
    assert loop.est.seen() == [("update", 1.0, 50.0, 0.0), ("miss", 1.03)]


def test_the_estimator_only_watches_the_mode_steers_on_the_measured_ball():
    loop, cam = cam_loop()
    cam.see(50.0, 1.0)
    rec = loop.tick(1.0, 0.01)
    assert rec.ball == cam.obs.ball
    assert rec.plate_cmd == make_mode("balance", loop.cfg)(rec.ball, rec.intent, loop.cfg)
    assert loop.snapshot()["est"] == [1.0, 2.0, 3.0, 4.0]
    loop.est.valid = False
    assert loop.snapshot()["est"] is None


def test_the_estimator_hears_the_sent_tilt_or_level():
    hand = Hand(x=1.0)
    loop = Loop(load(), hand)
    loop.mode, loop.est = "hand", Spy()
    hand.t = 1.0
    loop.tick(1.0, 0.01)
    assert loop.est.calls[-1] == ("command", 1.0, PlateCommand(0.0, 0.0))
    loop.arm()
    for i in range(1, 20):
        hand.t = 1.0 + i * 0.01
        rec = loop.tick(hand.t, 0.01)
    p = rec.plate_state
    assert rec.armed and p.sent_x_deg > 0
    assert loop.est.calls[-1] == ("command", hand.t, PlateCommand(p.sent_x_deg, p.sent_y_deg))


def drive(est):
    cam, hand = Cam(), Hand()
    loop = Loop(load(), hand, frames=cam)
    loop.mode, loop.est = "balance", est
    loop.arm()
    cmds = []
    for i in range(6):
        hand.t = t = 1.0 + i * 0.01
        cam.see(10.0 + i, t, valid=i != 3)
        cmds.append(loop.tick(t, 0.01).plate_cmd)
        loop.snapshot()
    return loop, cmds


@pytest.mark.parametrize("call", ["command", "update", "miss", "predict"])
def test_a_broken_estimator_switches_off_and_balance_carries_on(call, capsys):
    loop, cmds = drive(broken(call))
    assert cmds == drive(Spy())[1]
    assert loop.last.armed and loop.est is None and loop.snapshot()["est"] is None
    assert capsys.readouterr().out.count("estimator off") == 1


def test_closed_loop_names_real_modes():
    assert set(CLOSED_LOOP) <= set(MODES) and "hand" not in CLOSED_LOOP


def test_the_real_estimator_shows_up_in_the_snapshot_as_plain_numbers():
    cam = Cam()
    loop = Loop(load(), frames=cam)
    for i in range(5):
        cam.see(10.0 + i, 1.0 + i * 0.01)
        loop.tick(1.0 + i * 0.01, 0.01)
    est = json.loads(json.dumps(loop.snapshot()))["est"]
    assert len(est) == 4 and abs(est[0] - 14.0) < 5


@pytest.mark.parametrize("mode", ["balance", "path", "assist"])
def test_a_confirmed_exposure_mismatch_blocks_arming_the_closed_loop_modes(mode):
    loop = Loop(load(), exposure=MISMATCH)
    loop.mode = mode
    loop.arm()
    snap = loop.snapshot()
    assert not loop.safety.armed
    assert snap["arm_blocked"] == MISMATCH and snap["exposure"] == MISMATCH


def test_hand_mode_still_arms_with_a_mismatch_on_the_banner():
    loop = Loop(load(), exposure=MISMATCH)
    loop.mode = "hand"
    loop.arm()
    snap = loop.snapshot()
    assert loop.safety.armed and snap["arm_blocked"] is None and snap["exposure"] == MISMATCH


@pytest.mark.parametrize("exposure", [UNVERIFIED, None])
def test_an_unverified_or_clean_exposure_arms(exposure):
    loop = Loop(load(), exposure=exposure)
    loop.mode = "balance"
    loop.arm()
    assert loop.safety.armed and loop.snapshot()["arm_blocked"] is None


def test_a_class_mode_starts_fresh_on_arm_and_keeps_its_state_while_armed():
    hand = Hand()
    loop = Loop(load(), hand)
    loop.mode = "balance"
    before = loop.step
    hand.t = 1.0
    loop.tick(1.0, 0.01)
    assert loop.step is not before
    loop.arm()
    armed = loop.step
    for i in range(1, 4):
        hand.t = 1.0 + i * 0.01
        loop.tick(hand.t, 0.01)
    assert loop.step is armed
    loop.kill()
    loop.tick(hand.t + 0.01, 0.01)
    assert loop.step is not armed


def test_no_sim_ball_without_the_sim_plate():
    rec = Loop(load(), actuator=Board()).tick(1.0, 0.01)
    assert not rec.ball.valid
    assert Loop(load()).tick(1.0, 0.01).ball.valid


@pytest.mark.replay
@pytest.mark.skipif(not CLIP.exists(), reason="clip not on this machine")
def test_the_loop_tracks_a_replayed_clip_at_clip_time():
    cfg = load()
    stamps = [float(t) for t in CLIP.with_suffix(".csv").read_text().split()][START:END]
    loop = Loop(cfg, frames=ClipSource(CLIP, start=START, process=Pipeline(cfg).step))
    loop.mode = "balance"
    recs = [loop.tick(t, 0.01) for t in stamps]
    loop.frames.close()

    valid = [r.ball for r in recs if r.ball.valid]
    assert len(valid) >= 0.9 * len(stamps)
    assert all(r.ball.t_s == t for r, t in zip(recs, stamps, strict=True))
    half = cfg["rig"]["plate_inner_mm"] / 2
    assert all(abs(b.x_mm) < half and abs(b.y_mm) < half for b in valid)
    assert loop.snapshot()["est"] is not None
