import functools
import json
import math

import numpy as np
import pytest

from src.contracts import BallState
from src.control import mapping
from src.control.mapping import STRIDE_S, MappingWatch, angle_deg
from src.io.settings import ROOT, load

# hand play on the rig with the mapping right, 5 armed stretches
LOG = ROOT / "data" / "runs" / "20261108-025857.jsonl"

pytestmark = [pytest.mark.sim, pytest.mark.skipif(not LOG.exists(), reason="run log not here")]


@functools.cache
def stretches():
    out, cur = [], []
    for line in LOG.open():
        r = json.loads(line)
        if r["armed"]:
            cur.append(r)
        elif cur:
            out, cur = [*out, cur], []
    return [*out, cur] if cur else out


@pytest.fixture
def judged(monkeypatch):
    """Every direction the watcher judges: (window end, push direction, |angle| deg)."""
    seen, real = [], mapping.responses
    floor = load()["control"]["map_floor"]

    def spy(t, xy, e, excite):
        out = real(t, xy, e, excite)
        seen.extend((t[-1], d, angle_deg(d, r)) for d, r in out if math.hypot(*r) >= floor)
        return out

    monkeypatch.setattr(mapping, "responses", spy)
    return seen


def replay(rows, sign=(1, 1)):
    watch = MappingWatch(load())
    for r in rows:
        b, p = r["ball"], r["plate_state"]
        ball = BallState(b["x_mm"] * sign[0], b["y_mm"] * sign[1], 0.0, 0.0, b["valid"], b["t_s"])
        if watch.step(r["t_s"], ball, (p["sent_x_deg"], p["sent_y_deg"])):
            return r["ball"]["t_s"], watch.bad
    return None, None


def worst(judged):
    """Largest single |angle| and largest |angle| two windows in a row both pass, per axis."""
    single = pair = 0.0
    for k in range(2):
        on = {}
        for t, d, phi in judged:
            if abs(d[k]) >= 0.5:
                on[t] = max(on.get(t, 0.0), abs(phi))
        ts = sorted(on)
        single = max([single, *on.values()])
        for a, b in zip(ts[:-1], ts[1:], strict=True):
            if b - a <= 2 * STRIDE_S:
                pair = max(pair, min(on[a], on[b]))
    return single, pair


def test_the_right_mapping_on_the_rig_never_trips_with_room_to_spare(judged):
    c = load()["control"]
    for rows in stretches():
        assert replay(rows) == (None, None)
    single, pair = worst(judged)
    assert pair <= c["map_line_deg"] - 5 and single < c["map_flip_deg"], (single, pair)


def test_the_rig_pushes_where_the_run_log_fit_says(judged):
    # the run log fit puts servo x at about 18 deg and servo y at about 116 deg in plate mm
    for rows in stretches():
        replay(rows)
    for k, axis_deg in ((0, 18), (1, 116 - 90)):
        phis = [phi for _, d, phi in judged if abs(d[k]) >= 0.9]
        assert abs(np.median(phis) - axis_deg) < 25, ("xy"[k], np.median(phis))


@pytest.mark.parametrize("sign, axis", [((-1, 1), "x"), ((1, -1), "y")])
def test_a_mirrored_camera_axis_on_the_rig_trips_within_a_few_windows(judged, sign, axis):
    k, lags = "xy".index(axis), []
    for rows in stretches():
        judged.clear()
        t, bad = replay(rows, sign)
        # a stretch that doesn't trip judges this axis once at most, too little push along it
        if t is None:
            continue
        assert axis in bad
        on = sorted({w for w, d, _ in judged if abs(d[k]) >= 0.5})
        assert len(on) <= 4, (len(on), bad)
        lags.append(round(float(t - on[0]), 2))
    print(f"mirrored {axis}: trips {lags} s after the first excited window")
    assert len(lags) >= 3 and max(lags) <= 3 * STRIDE_S + 0.01, lags
