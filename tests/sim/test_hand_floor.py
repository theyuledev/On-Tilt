import copy
import math
from collections import deque

import pytest

from src.control import ROLL
from src.inputs.shaping import shape
from src.io.settings import load
from src.safety import limits

pytestmark = pytest.mark.sim

DT = 0.01
US_PER_DEG = 100.0  # the real plate, about 1.5 deg at the clamp
PLATE_DELAY_S = 0.10
REACT_S = 0.2
KP = 0.012  # intent per mm
KD = 0.010  # intent per mm/s
RIM_MM = 140.0
START_MM, GOAL_MM, BAND_MM = -80.0, 80.0, 10.0

REAL = load()
X_LO, X_HI, _, _ = limits(REAL["rig"])
CLAMP_DEG = (X_HI - X_LO) / 2 / US_PER_DEG
FRICTION = REAL["control"]["rolling_friction_x_mm_s2"]


def shaping(cap: float, us_per_deg: float, friction: bool) -> dict:
    cfg = copy.deepcopy(REAL)
    cfg["input"]["intent_gain_deg"] = cap
    cfg["control"].pop("player_max_deg", None)
    cfg["rig"]["us_per_deg_x"] = us_per_deg
    if not friction:
        cfg["control"]["rolling_friction_x_mm_s2"] = 0.0
    return cfg


NEW = shaping(6.0, US_PER_DEG, friction=True)
# gain 6 and no floor, with the clamp out of the way
TODAY = shaping(6.0, 1.0, friction=False)


def roll(x: float, v: float, tilt_deg: float) -> tuple[float, float]:
    a = ROLL * REAL["control"]["sim_g_mm_s2"] * math.sin(math.radians(tilt_deg))
    if v == 0.0 and abs(a) <= FRICTION:
        return x, 0.0
    nv = v + (a - math.copysign(FRICTION, v or a)) * DT
    if v and nv * v < 0 and abs(a) <= FRICTION:
        nv = 0.0
    return x + nv * DT, nv


def player(seen: tuple[float, float], kp: float, kd: float) -> float:
    x, v = seen
    return max(-1.0, min(1.0, kp * (GOAL_MM - x) - kd * v))


def run(cfg: dict, hold=None, x0=START_MM, seconds=6.0, kp=KP, kd=KD) -> list[float]:
    """Delayed player -> shape -> delayed, clamped plate -> ball with rolling friction, x only."""
    seen = deque([(x0, 0.0)] * round(REACT_S / DT))
    plate = deque([0.0] * round(PLATE_DELAY_S / DT))
    x, v, xs = x0, 0.0, []
    for _ in range(round(seconds / DT)):
        intent = player(seen.popleft(), kp, kd) if hold is None else hold
        plate.append(shape(intent, 0.0, cfg)[0])
        x, v = roll(x, v, max(-CLAMP_DEG, min(CLAMP_DEG, plate.popleft())))
        seen.append((x, v))
        xs.append(x)
    return xs


def settle_s(xs: list[float]) -> float:
    out = [i for i, x in enumerate(xs) if abs(x - GOAL_MM) > BAND_MM]
    if out and out[-1] == len(xs) - 1:
        return math.inf
    return (out[-1] + 1) * DT if out else 0.0


def overshoot_mm(xs: list[float]) -> float:
    return max(0.0, max(xs) - GOAL_MM)


def test_full_stick_is_the_plate_clamp_not_past_it():
    assert shape(1.0, 0.0, NEW)[0] == pytest.approx(CLAMP_DEG)


def test_just_past_twice_the_dead_zone_starts_a_resting_ball():
    nudge = 2.02 * NEW["input"]["dead_zone"]
    assert run(NEW, hold=nudge, x0=0.0, seconds=3.0)[-1] > 0.0
    assert run(TODAY, hold=nudge, x0=0.0, seconds=3.0)[-1] == 0.0


def test_inside_the_dead_zone_leaves_a_resting_ball_alone():
    still = 0.9 * NEW["input"]["dead_zone"]
    assert set(run(NEW, hold=still, x0=0.0, seconds=3.0)) == {0.0}


def test_corner_to_corner_settles_quick_without_hitting_the_rim():
    xs = run(NEW)
    assert settle_s(xs) < 3.0
    assert overshoot_mm(xs) < BAND_MM
    assert max(abs(x) for x in xs) < RIM_MM
