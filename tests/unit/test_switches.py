from dataclasses import asdict

import pytest

from src.contracts import Choices, Intent
from src.io.settings import load
from src.loop import Loop
from src.modes import Balance
from src.switches import choose, parse, view

pytestmark = pytest.mark.unit

ALL = Choices(mode="path", steer="keys", hand_mode="palm", guard=True)


class Hand:
    seen = calibrating = False
    progress = 0.0
    mode = None

    def __init__(self):
        self.intent = Intent(0.5, 0.0, True, 10.0)

    def latest(self):
        return self.intent

    def set_mode(self, mode):
        self.mode = mode


def test_parse_keeps_what_the_page_may_pick_and_refuses_the_rest():
    assert parse({"mode": "balance", "x": 1}) == Choices(mode="balance")
    assert parse({"steer": "keys", "hand_mode": "palm", "guard": False}) == Choices(
        steer="keys", hand_mode="palm", guard=False
    )
    assert parse({"mode": None}) == Choices()
    bad = ({"mode": "assist"}, {"mode": ["hand"]}, {"steer": "mouse"}, {"hand_mode": "fist"})
    for raw in (*bad, {"guard": 1}, {"guard": "on"}):
        with pytest.raises(ValueError):
            parse(raw)


def test_snapshot_carries_the_choices_from_settings():
    cfg = load()
    plain = {
        "mode": cfg["control"]["mode"],
        "hand_mode": cfg["input"]["hand_mode"],
        "guard": cfg["control"]["guard_on"],
    }
    assert Loop(cfg).snapshot()["choices"] == {**plain, "steer": "keys"}
    assert Loop(cfg, Hand()).snapshot()["choices"] == {**plain, "steer": "hand"}


def test_a_disarmed_loop_takes_the_mode_and_builds_it():
    loop = Loop(load())
    choose(loop, Choices(mode="balance"))
    assert loop.mode == loop.cfg["control"]["mode"] == "balance" and isinstance(loop.step, Balance)
    assert view(loop).mode == "balance"
    loop.arm()
    assert loop.tick(1.0, 0.01).mode == "balance"


def test_an_armed_or_killing_loop_keeps_every_choice():
    loop = Loop(load(), Hand())
    loop.cfg["control"]["guard_on"] = False
    before = view(loop)
    loop.arm()
    choose(loop, ALL)
    loop.kill()
    assert loop.safety.armed and loop.safety.killing
    choose(loop, ALL)
    assert view(loop) == before and loop.mode == "hand" and loop.hand.mode is None


def test_keys_only_ignores_a_live_hand_until_hand_is_picked_again():
    loop = Loop(load(), Hand())
    loop.keys._intent = Intent(-1.0, 0.0, True, 10.0)
    assert loop.tick(10.01, 0.01).intent.x == 0.5
    choose(loop, Choices(steer="keys"))
    assert view(loop).steer == "keys" and loop.tick(10.02, 0.01).intent.x == -1.0
    choose(loop, Choices(steer="hand"))
    assert view(loop).steer == "hand" and loop.tick(10.03, 0.01).intent.x == 0.5


def test_steering_is_keys_whatever_is_picked_when_no_hand_is_wired():
    loop = Loop(load())
    choose(loop, Choices(steer="hand", hand_mode="palm"))
    assert asdict(view(loop))["steer"] == "keys"
    assert view(loop).hand_mode == loop.cfg["input"]["hand_mode"]


def test_hand_mode_goes_to_the_hand_source_and_the_view_shows_what_it_runs():
    loop = Loop(load(), Hand())
    assert view(loop).hand_mode == loop.cfg["input"]["hand_mode"]
    choose(loop, Choices(hand_mode="palm"))
    assert loop.hand.mode == "palm" and view(loop).hand_mode == "palm"


def test_guard_only_switches_on_a_build_that_has_one():
    loop = Loop(load())
    choose(loop, Choices(guard=True))
    assert view(loop).guard is True and loop.cfg["control"]["guard_on"] is True
    del loop.cfg["control"]["guard_on"]
    choose(loop, Choices(guard=False))
    assert view(loop).guard is None and "guard_on" not in loop.cfg["control"]


def test_a_mode_that_cant_be_built_changes_nothing():
    cfg = load()
    cfg["control"]["path_mm"] = []
    loop = Loop(cfg)
    with pytest.raises(ValueError):
        choose(loop, Choices(mode="path", steer="keys"))
    assert loop.mode == cfg["control"]["mode"] == "hand" and not loop.keys_only
    loop.tick(1.0, 0.01)
