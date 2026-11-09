import json
import re

import pytest

from src.games import race
from src.games.progress import Progress
from src.io.settings import ROOT
from tests.unit.test_race import GAME, rec
from tests.unit.test_web import RACE_FIELDS

pytestmark = pytest.mark.unit

STATIC = ROOT / "src" / "web" / "static"
# names that hold the race view in each file; r in race.js is the ball radius
NAMES = {
    "hud.js": ("r", "was", "prev", "s.race"),
    "race.js": ("race", "s.race"),
    "results.js": ("r", "s.race"),
}
TRACE = [[0.0, -60.0, 0.0], [2.0, 40.0, 0.0]]
SPLITS = [1.0, 2.0, 3.0]


def races():
    lap = dict(go_s=3.0, t0_s=4.0, last_valid_t=6.0)
    part = Progress(reached=1, off_s=1.5, splits=(5.0,), prev_t=6.0)
    done = Progress(reached=3, off_s=1.5, splits=(5.0, 6.0, 7.0), done_t=7.0, prev_t=7.0)
    return {
        "idle": race.Race(),
        "calibrating": race.Race("calibrating", last_valid_t=1.0, cal={"state": "trimming"}),
        "staging": race.Race("staging", last_valid_t=1.0),
        "countdown": race.Race("countdown", go_s=3.0, last_valid_t=1.0),
        "running": race.Race("running", progress=part, **lap),
        "finished": race.Race("finished", progress=done, **lap),
        "dnf": race.Race("dnf", progress=part, reason="killed", end_s=6.0, **lap),
    }


VIEWS = {s: race.view(r, rec(6.0), GAME, TRACE, SPLITS) for s, r in races().items()}
DICTS = {k for v in VIEWS.values() for k, x in v.items() if isinstance(x, dict)}


def page_reads():
    top, nested = set(), set()
    for name, names in NAMES.items():
        js = re.sub(r"//.*", "", (STATIC / name).read_text())
        alts = "|".join(re.escape(n) for n in names)
        for field, sub in re.findall(rf"(?<![\w.])(?:{alts})\.(\w+)(?:\.(\w+))?", js):
            top.add(field)
            # r.elapsed_s.toFixed is a method call, not a nested field
            if sub and field in DICTS:
                nested.add((field, sub))
    return top, nested


def test_the_page_reads_the_race_fields_we_expect():
    top, nested = page_reads()
    assert {"state", "reached", "elapsed_s", "ghost_mm", "ghost_lap_s", "score"} <= top
    assert {("score", "time_s"), ("score", "penalty_s")} <= nested


@pytest.mark.parametrize("state", list(VIEWS))
def test_every_race_field_the_page_reads_is_in_the_view(state):
    v = VIEWS[state]
    top, nested = page_reads()
    assert v["state"] == state
    assert top <= set(v), top - set(v)
    assert set(RACE_FIELDS) <= set(v)
    for field, sub in nested:
        if isinstance(v[field], dict):
            assert sub in v[field], f"{field}.{sub}"
    assert json.loads(json.dumps(v))["state"] == state


def test_the_finished_score_has_every_field_the_page_reads():
    score = VIEWS["finished"]["score"]
    assert isinstance(score, dict)
    assert {sub for field, sub in page_reads()[1] if field == "score"} <= set(score)
