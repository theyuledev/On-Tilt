import pytest

from src.contracts import ServoCommand
from src.sim.plate import SimActuator

pytestmark = pytest.mark.sim

# 10 us per degree so expected tilts are easy to work out by hand
RIG = {
    "center_x_us": 1500,
    "center_y_us": 1500,
    "trim_x_us": 0,
    "trim_y_us": 0,
    "us_per_deg_x": 10.0,
    "us_per_deg_y": 10.0,
}
SERVO = {
    "servo_dead_time_s": 0.02,
    "servo_tau_s": 0.05,
    "servo_top_speed_us_s": 1000.0,
    "servo_dead_band_us": 10,
}
DT = 0.001


def run(act, x_us, y_us=1500, t0=0.0, seconds=0.5):
    cmd = ServoCommand(x_us, y_us)
    return [act.send(cmd, t0 + i * DT) for i in range(round(seconds / DT))]


def at(states, t):
    return states[round(t / DT)]


def test_small_step_waits_dead_time_then_lags_first_order():
    states = run(SimActuator(RIG, SERVO), 1540, 1460)

    assert states[0].sent_x_deg == 4.0
    assert at(states, 0.019).est_x_deg == 0
    # 40 us step is under the speed cap: 63% of it one tau after the dead time
    assert at(states, 0.07).est_x_deg == pytest.approx(2.53, abs=0.1)
    assert at(states, 0.07).est_y_deg == pytest.approx(-2.53, abs=0.1)
    assert states[-1].est_x_deg == pytest.approx(4.0)


def test_big_step_is_capped_at_top_speed():
    states = run(SimActuator(RIG, SERVO), 1800, seconds=0.6)

    # 1000 us/s for 0.1 s is 100 us, 10 deg at this rig scale
    rise = at(states, 0.15).est_x_deg - at(states, 0.05).est_x_deg
    assert rise == pytest.approx(10.0, abs=0.2)
    assert states[-1].est_x_deg == pytest.approx(30.0)


def test_pulse_change_under_dead_band_does_nothing():
    act = SimActuator(RIG, SERVO)

    assert all(s.est_x_deg == 0 for s in run(act, 1509))
    assert run(act, 1510, t0=0.5)[-1].est_x_deg == pytest.approx(1.0)
    assert all(s.est_x_deg == pytest.approx(1.0) for s in run(act, 1519, t0=1.0))
