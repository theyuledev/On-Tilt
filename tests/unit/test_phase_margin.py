import math

import pytest

from src.control.margin import phase_margin_deg

pytestmark = pytest.mark.unit

G = 9810
KP, KD = 0.07, 0.04


def crossover(kp, kd):
    # |kp + j kd w| k / w^2 = 1, solved for w
    k = 5 / 7 * G * math.pi / 180
    return math.sqrt((k * k * kd * kd + math.hypot(k * k * kd * kd, 2 * k * kp)) / 2)


def test_p_only_on_a_double_integrator_has_no_margin():
    assert phase_margin_deg(KP, 0.0, 0.0, 0.0, G) == pytest.approx(0.0, abs=0.05)


def test_pd_without_delay_gets_the_textbook_lead():
    wc = crossover(KP, KD)
    lead = math.degrees(math.atan2(KD * wc, KP))
    assert phase_margin_deg(KP, KD, 0.0, 0.0, G) == pytest.approx(lead, abs=0.1)


def test_delay_costs_wc_times_t_and_a_big_one_goes_unstable():
    wc = crossover(KP, KD)
    clean = phase_margin_deg(KP, KD, 0.0, 0.0, G)
    lost = math.degrees(wc * 0.05)
    assert phase_margin_deg(KP, KD, 0.05, 0.0, G) == pytest.approx(clean - lost, abs=0.1)
    assert phase_margin_deg(KP, KD, 0.5, 0.0, G) < 0


def test_no_integral_leaves_the_pd_margin_alone():
    pd = phase_margin_deg(KP, KD, 0.05, 0.03, G)
    assert phase_margin_deg(KP, KD, 0.05, 0.03, G, ki=0.0) == pd


def test_an_integral_term_costs_a_little_lead():
    ki = 0.02
    wc = crossover(KP, KD)
    lost = phase_margin_deg(KP, KD, 0.0, 0.0, G) - phase_margin_deg(KP, KD, 0.0, 0.0, G, ki=ki)
    direct = math.degrees(math.atan(KD * wc / KP) - math.atan((KD * wc - ki / wc) / KP))
    # ki also pulls the crossover down a little, which costs some more lead on top
    assert direct < lost < 2 * direct
