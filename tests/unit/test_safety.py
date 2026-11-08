import pytest

from src.contracts import ServoCommand
from src.io.serial_link import ENVELOPE_US
from src.io.settings import load
from src.safety import LEVEL_HOLD_S, SafetyState, limits, step

pytestmark = pytest.mark.unit

RIG = {
    "center_x_us": 1500,
    "center_y_us": 1500,
    "trim_x_us": 0,
    "trim_y_us": 0,
    "limit_x_us": 400,
    "limit_y_us": 400,
    "startup_range_us": 150,
    "startup_flag": True,
    "slew_us_s": 1000,
    "kill_ramp_us_s": 100,
}


def armed():
    s = SafetyState()
    s.arm()
    return s


def run(s, target, n, dt=0.01, stale=False, rig=RIG):
    out = None
    for _ in range(n):
        out = step(s, target, stale, dt, rig)
    return out


def test_sends_nothing_unless_armed():
    assert step(SafetyState(), ServoCommand(1600, 1600), False, 0.01, RIG) is None


def test_first_command_is_center():
    out = step(armed(), ServoCommand(1500, 1500), False, 0.01, RIG)
    assert out == ServoCommand(1500, 1500)


def test_clamps_to_startup_range_while_flag_on():
    out = run(armed(), ServoCommand(2400, 600), 200)
    assert out == ServoCommand(1650, 1350)


def test_clamps_to_limits_once_flag_cleared():
    rig = {**RIG, "startup_flag": False}
    out = run(armed(), ServoCommand(2400, 600), 200, rig=rig)
    assert out == ServoCommand(1900, 1100)


def test_slew_limits_each_step():
    out = step(armed(), ServoCommand(1650, 1500), False, 0.01, RIG)
    assert out == ServoCommand(1510, 1500)


def test_stale_intent_ramps_to_level_and_stays_armed():
    s = armed()
    run(s, ServoCommand(1650, 1650), 50)
    out = step(s, ServoCommand(1650, 1650), True, 0.01, RIG)
    assert out == ServoCommand(1649, 1649)
    out = run(s, ServoCommand(1650, 1650), 300, stale=True)
    assert out == ServoCommand(1500, 1500)
    assert s.armed


def test_kill_ramps_to_level_then_disarms():
    s = armed()
    run(s, ServoCommand(1650, 1650), 50)
    s.kill()
    out = step(s, ServoCommand(1650, 1650), False, 0.01, RIG)
    assert out == ServoCommand(1649, 1649)
    run(s, ServoCommand(1650, 1650), 300)
    assert not s.armed
    assert step(s, ServoCommand(1650, 1650), False, 0.01, RIG) is None


def test_kill_holds_the_level_pulse_40ms_before_disarming():
    s = armed()
    run(s, ServoCommand(1650, 1650), 50)
    s.kill()
    run(s, ServoCommand(1650, 1650), 148)
    assert step(s, ServoCommand(1650, 1650), False, 0.01, RIG) == ServoCommand(1501, 1501)
    level = []
    while (out := step(s, ServoCommand(1650, 1650), False, 0.01, RIG)) is not None:
        level.append(out)
    assert level == [ServoCommand(1500, 1500)] * 5
    assert not s.armed and s.level_s == 0


def test_kill_hold_starts_after_a_slow_tick_reaches_level():
    s = armed()
    run(s, ServoCommand(1650, 1650), 50)
    s.kill()
    run(s, ServoCommand(1650, 1650), 149)
    dts = [0.05] + [0.01] * 10
    out = [step(s, ServoCommand(1650, 1650), False, dt, RIG) for dt in dts]
    level = out[: out.index(None)]
    assert level == [ServoCommand(1500, 1500)] * 5
    # disarms on the last level pulse, a full hold after the first one
    assert sum(dts[1 : len(level)]) >= LEVEL_HOLD_S
    assert not s.armed


def test_arm_during_kill_ramp_is_ignored():
    s = armed()
    run(s, ServoCommand(1650, 1650), 50)
    s.kill()
    run(s, ServoCommand(1650, 1650), 10)
    s.arm()
    assert s.killing
    assert step(s, ServoCommand(1650, 1650), False, 0.01, RIG) == ServoCommand(1639, 1639)
    run(s, ServoCommand(1650, 1650), 300)
    assert not s.armed and not s.killing
    s.arm()
    assert s.armed


@pytest.mark.parametrize("startup_flag", [False, True])
def test_settings_limits_stay_inside_the_firmware_envelope(startup_flag):
    rig = {**load()["rig"], "startup_flag": startup_flag}
    lo, hi = ENVELOPE_US
    for edge in limits(rig):
        assert lo <= edge <= hi
