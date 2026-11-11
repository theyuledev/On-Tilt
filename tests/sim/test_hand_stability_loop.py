from types import SimpleNamespace

import pytest

from src.contracts import Intent
from src.inputs.hand import HandIntent
from src.inputs.prediction import HandPredictor
from src.io.settings import BASE, load
from src.loop import Loop
from src.safety import limits

pytestmark = pytest.mark.sim


def run(spike=False):
    cfg = load(local=BASE)
    cfg["control"]["mode"] = "hand"
    inp = cfg["input"]
    span = inp["hand_tilt_span_deg"]
    hand = HandIntent(inp)
    prediction = HandPredictor()
    feed = SimpleNamespace(seen=True, calibrating=False, progress=1.0)
    intent = Intent(0.0, 0.0, False, 0.0)
    feed.latest = lambda: intent
    loop = Loop(cfg, hand=feed)
    loop.arm()
    frame = 0
    records = []
    for tick in range(400):
        now = tick / 100
        if now >= frame / 30 + 0.025:
            at = (0.0, 0.0) if frame < 35 else (0.5 * span, -0.3 * span)
            if spike and frame == 75:
                at = (-span, span)
            measured = hand.update(at, frame / 30, recentre=frame == 0)
            intent = prediction.update(measured, now, recentre=frame == 0)
            frame += 1
        records.append(loop.tick(now, 0.01))
    return cfg, records


def test_a_pose_spike_at_webcam_rate_cannot_kick_the_faster_control_loop():
    cfg, steady = run()
    _, noisy = run(spike=True)
    assert [r.servo_cmd for r in noisy] == [r.servo_cmd for r in steady]
    x_lo, x_hi, y_lo, y_hi = limits(cfg["rig"])
    assert all(x_lo <= r.servo_cmd.x_us <= x_hi for r in noisy)
    assert all(y_lo <= r.servo_cmd.y_us <= y_hi for r in noisy)
    assert noisy[-1].intent.x > 0 and noisy[-1].intent.y > 0
