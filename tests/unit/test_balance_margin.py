import pytest

from src.control.margin import balance_margin
from tools import margin_check

pytestmark = pytest.mark.unit


def cfg(kp, kd, ki, dead=0.046, fps=120, ema=0.0):
    control = {
        "sim_tick_hz": 100,
        "servo_dead_time_s": dead,
        "vel_ema_s": ema,
        "servo_tau_s": 0.035,
        "sim_g_mm_s2": 9810,
        "kp_deg_mm": kp,
        "kd_deg_s_mm": kd,
        "ki_deg_s_mm": ki,
    }
    return {"control": control, "rig": {"camera_fps": fps}}


def test_delay_is_dead_time_plus_a_tick_plus_half_the_velocity_window():
    m = balance_margin(cfg(0.05, 0.0375, 0.02))
    assert m["tick_s"] == pytest.approx(0.01)
    # camera beats the loop, so 4 ticks between the two fixes, lagging 2
    assert m["vel_lag_s"] == pytest.approx(0.02)
    assert m["delay_s"] == pytest.approx(0.076)


def test_a_camera_slower_than_the_loop_sets_the_velocity_window():
    assert balance_margin(cfg(0.05, 0.0375, 0.02, fps=50))["vel_lag_s"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("kp", "kd", "ki", "dead", "deg", "ok"),
    [
        (0.05, 0.0375, 0.02, 0.046, 44.25, False),
        (0.035, 0.02625, 0.014, 0.046, 46.02, True),
        (0.035, 0.03, 0.01, 0.046, 48.55, True),
        (0.035, 0.02625, 0.014, 0.065, 42.43, False),
    ],
)
def test_margin_and_verdict(kp, kd, ki, dead, deg, ok):
    m = balance_margin(cfg(kp, kd, ki, dead))
    assert m["margin_deg"] == pytest.approx(deg, abs=0.05)
    assert m["pass"] is ok


def test_a_velocity_filter_adds_its_time_constant_to_the_delay():
    bare = balance_margin(cfg(0.035, 0.02625, 0.014))
    smoothed = balance_margin(cfg(0.035, 0.02625, 0.014, ema=0.03))
    assert smoothed["delay_s"] - bare["delay_s"] == pytest.approx(0.03)
    assert smoothed["margin_deg"] < bare["margin_deg"]


def test_ring_period_comes_from_the_crossover():
    m = balance_margin(cfg(0.035, 0.02625, 0.014))
    assert m["crossover_rad_s"] == pytest.approx(3.296, abs=0.01)
    assert m["ring_period_s"] == pytest.approx(1.906, abs=0.01)


def test_script_prints_each_number_then_the_verdict(monkeypatch, capsys):
    monkeypatch.setattr(margin_check, "load", lambda: cfg(0.05, 0.0375, 0.02))
    margin_check.main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "FAIL, under 45 deg"
    assert any(line.split() == ["delay_s", "0.076"] for line in lines)
