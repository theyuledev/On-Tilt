import shutil

import pytest

from tests.unit.page import race, run, snap

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

OPEN = ("open", None)
GHOST = {"ghost_lap_s": 31.5}
SCORE = {"time_s": 31.19, "penalty_s": 2.0, "checkpoints": 4}
DONE = race("finished", elapsed_s=29.19, reached=4, delta_s=-0.31, score=SCORE, **GHOST)


def show(*snaps, **kw):
    return run([OPEN, *[("snap", s) for s in snaps]], **kw)["page"]


def lamps(page):
    return [(k["text"], k["cls"]) for k in page["cps"]["kids"]]


def pressed(page, group):
    return [k["attrs"]["aria-pressed"] for k in page[group]["kids"]]


def test_ready_page_shows_a_dim_timer_the_first_gate_next_and_the_ball_in_mm():
    ball = {
        "x_mm": -112.54,
        "y_mm": 112.46,
        "vx_mm_s": 0.0,
        "vy_mm_s": 0.0,
        "valid": True,
        "t_s": 1,
    }
    p = show(snap(ball=ball, race=race("idle", **GHOST)))
    assert (p["timelabel"]["text"], p["timer"]["text"], p["timer"]["cls"]) == (
        "TIME",
        "00:00.00",
        "idle",
    )
    assert (p["best"]["text"], p["ghostsay"]["text"], p["delta"]["text"]) == (
        "00:31.50",
        "VS GHOST",
        "--",
    )
    assert lamps(p) == [("1", "lamp next"), ("2", "lamp"), ("3", "lamp"), ("F", "lamp")]
    assert (p["ballx"]["text"], p["bally"]["text"]) == ("X-112.5", "Y+112.5")
    assert (p["tiltx"]["text"], p["tilty"]["text"]) == ("X +0.00°", "Y +0.00°")
    assert p["scale"]["text"] == "1 px = 0.42 mm · 120 fps"
    assert p["sub"]["text"] == "COURSE LOADED · 4 GATES"
    assert (p["armed"]["text"], p["armed"]["cls"].strip()) == ("DISARMED", "chip")
    assert (p["banner"]["text"], p["banner"]["data"]["tone"]) == ("click arm to start", "")
    assert (
        p["racebtn"]["text"] == "RACE" and not p["arm"]["disabled"] and not p["racebtn"]["disabled"]
    )
    assert not p["scores"]["hidden"] and not p["gates"]["hidden"] and p["nomaze"]["hidden"]


def test_lights_follow_the_snapshot():
    p = show(snap())
    got = {
        i: (p[i]["text"], p[i]["cls"].strip())
        for i in ("link", "board", "camlight", "ballpill", "hand")
    }
    assert got == {
        "link": ("LINK", "light on"),
        "board": ("BOARD", "light on"),
        "camlight": ("NO CAM", "light"),
        "ballpill": ("BALL", "light on"),
        "hand": ("NO HAND", "light"),
    }
    assert p["stale"]["hidden"] and p["level"]["hidden"]
    lost = {"valid": False, "x_mm": 0, "y_mm": 0, "vx_mm_s": 0, "vy_mm_s": 0, "t_s": 0}
    p = show(snap(armed=True, stale=True, board_lost=True, camera_lost=True, hand_seen=True,
                  hand_calibrating=False, ball=lost))  # fmt: skip
    got = {i: (p[i]["text"], p[i]["cls"]) for i in ("board", "camlight", "ballpill", "hand")}
    assert got == {
        "board": ("BOARD LOST", "light warn"),
        "camlight": ("CAM LOST", "light warn"),
        "ballpill": ("NO BALL", "light warn"),
        "hand": ("HAND", "light on"),
    }
    assert not p["stale"]["hidden"] and not p["level"]["hidden"]
    assert (p["ballx"]["text"], p["bally"]["text"]) == ("X--.-", "Y--.-")
    assert show(snap(hand_seen=True, hand_calibrating=True))["hand"]["text"] == "HOLD STILL"


def test_a_maze_with_walls_is_named_by_its_strokes_and_no_maze_says_so():
    walls = [[[0, 0], [10, 0]], [[0, 5], [10, 5]]]
    maze = {"points": [[0, 0], [10, 0]], "checkpoints": [1], "start": [0, 0], "walls": walls}
    assert show(snap(), race_maze=maze)["sub"]["text"] == "MAZE READ FROM BOARD · 2 STROKES"
    assert show(snap(race=...), race_maze=None)["sub"]["text"] == "NO MAZE ON THE BOARD"


def test_racing_lights_reached_gates_runs_the_timer_and_says_who_is_ahead():
    live = {"elapsed_s": 14.2, "reached": 2, **GHOST}
    p = show(snap(armed=True, race=race("running", delta_s=-0.42, **live)))
    assert (p["timer"]["text"], p["timer"]["cls"]) == ("00:14.20", "")
    assert lamps(p) == [("1", "lamp hit"), ("2", "lamp hit"), ("3", "lamp next"), ("F", "lamp")]
    assert (p["ghostsay"]["text"], p["delta"]["text"]) == ("AHEAD OF GHOST", "-0.42")
    assert p["ghostrow"]["cls"] == "score ahead"
    assert (p["armed"]["text"], p["armed"]["cls"]) == ("ARMED", "chip on")
    assert p["racebtn"]["text"] == "ABORT" and p["arm"]["disabled"]
    assert p["banner"]["text"] == "gate 3 is next"
    p = show(snap(armed=True, race=race("running", delta_s=0.8, **live)))
    assert (p["ghostsay"]["text"], p["delta"]["text"]) == ("BEHIND GHOST", "+0.80")
    assert p["ghostrow"]["cls"] == "score behind"
    p = show(snap(armed=True, race=race("running", elapsed_s=3.0, reached=1)))
    assert (p["best"]["text"], p["ghostsay"]["text"], p["delta"]["text"]) == (
        "--:--.--",
        "FIRST LAP",
        "--",
    )


def test_no_race_text_while_staging_and_the_timer_stays_dim():
    p = show(
        snap(armed=True, race=race("running", elapsed_s=3.0, reached=1)),
        snap(armed=True, race=race("staging")),
    )
    assert [p[i]["text"] for i in ("count", "split", "pen")] == ["", "", ""]
    assert p["result"]["hidden"] and (p["timer"]["text"], p["timer"]["cls"]) == ("00:00.00", "idle")
    assert lamps(p)[0] == ("1", "lamp next") and p["racebtn"]["text"] == "ABORT"
    assert p["banner"]["text"] == "roll the ball into the start ring"


def test_overlay_texts_pop_over_the_board():
    count = race("countdown", elapsed_s=-2.4, **GHOST)
    p = show(snap(armed=True, race=count))
    assert (p["count"]["text"], p["count"]["cls"]) == ("3", "hud tick")
    go = race("running", elapsed_s=0.0, **GHOST)
    p = show(snap(armed=True, race=count), snap(armed=True, race=go))
    assert (p["count"]["text"], p["count"]["cls"]) == ("GO", "hud go")
    gate = race("running", elapsed_s=5.0, reached=1, delta_s=0.25, penalty_s=1.5, **GHOST)
    p = show(snap(armed=True, race=go), snap(armed=True, race=gate))
    assert (p["split"]["text"], p["split"]["cls"], p["split"]["style"]["color"]) == (
        "+0.25",
        "hud split",
        "#E69F00",
    )
    assert (p["pen"]["text"], p["pen"]["cls"]) == ("+1.5 s", "hud pen")
    first = race("running", elapsed_s=5.0, reached=1)
    p = show(snap(armed=True, race=race("running", elapsed_s=0.0)), snap(armed=True, race=first))
    assert p["split"]["text"] == "CP 1  5.00"
    dnf = race("dnf", elapsed_s=5.2, reached=1, reason="ball lost", **GHOST)
    p = show(snap(armed=True, race=gate), snap(armed=True, race=dnf))
    assert [k["text"] for k in p["result"]["kids"]] == ["DNF", "ball lost\narm, then R"]
    assert p["result"]["cls"] == "hud end" and not p["result"]["hidden"]
    p = show(
        snap(armed=True, race=count), snap(armed=True, race=race("idle", reason="left the start"))
    )
    assert p["result"]["kids"][0]["text"] == "false start" and p["result"]["cls"] == "hud brief"
    p = show(snap(armed=True, race=gate), snap(armed=True, race=DONE))
    assert [k["text"] for k in p["result"]["kids"]] == ["FINISH", "00:31.19"]
    assert p["result"]["cls"] == "hud brief"


def test_off_path_flags_the_arena():
    off = race("running", elapsed_s=1.0, off_path=True)
    assert run([OPEN, ("snap", snap(armed=True, race=off))])["body"] == "offpath"
    assert (
        run([OPEN, ("snap", snap(armed=True, race=race("running", elapsed_s=1.0)))])["body"] == ""
    )


def test_finish_shows_the_scored_time_then_the_name_entry_and_enter_saves_once():
    steps = [OPEN, ("snap", snap(race=DONE))]
    p = run(steps)["page"]
    assert (p["timelabel"]["text"], p["timer"]["text"]) == ("FINISH TIME", "00:31.19")
    assert (p["ghostsay"]["text"], p["delta"]["text"]) == ("BEAT THE GHOST BY", "-0.31")
    assert all(cls == "lamp hit" for _, cls in lamps(p))
    assert p["banner"]["text"] == "you beat the ghost!"
    # the pop over the board gets a second first
    assert p["results"]["hidden"] and not p["choices"]["hidden"] and p["resname"]["disabled"]
    steps += [("wait", 1000), ("snap", snap(race=DONE))]
    p = run(steps)["page"]
    assert not p["results"]["hidden"] and p["choices"]["hidden"]
    assert p["resname"]["value"] == "JUDGE 1" and not p["resname"]["disabled"]
    assert p["respen"]["text"] == "Penalty +2.0 s." and p["ressaved"]["hidden"]
    steps += [
        ("down", ["Enter", "resname"]),
        ("frame", None),
        ("snap", snap(race=DONE)),
        ("frame", None),
    ]
    out = run([*steps, ("down", ["Enter", "resname"]), ("click", "ressave"), ("frame", None)])
    assert [m["save"] for m in out["sent"] if "save" in m] == ["JUDGE 1"]
    p = out["page"]
    assert not p["ressaved"]["hidden"] and p["resname"]["disabled"] and p["ressave"]["disabled"]


def test_save_button_saves_the_typed_name():
    steps = [OPEN, ("snap", snap(race=DONE)), ("wait", 1000), ("snap", snap(race=DONE))]
    out = run([*steps, ("type", ["resname", " ada "]), ("click", "ressave"), ("frame", None)])
    assert [m["save"] for m in out["sent"] if "save" in m] == ["ada"]
    out = run([*steps, ("type", ["resname", "  "]), ("click", "ressave"), ("frame", None)])
    assert not [m for m in out["sent"] if "save" in m]


def test_name_field_only_takes_keys_while_disarmed():
    armed = snap(armed=True, race=DONE)
    steps = [OPEN, ("snap", armed), ("wait", 1000), ("snap", armed)]
    out = run([*steps, ("click", "ressave"), ("down", ["Space", "resname"]), ("frame", None)])
    p = out["page"]
    assert not p["results"]["hidden"] and p["resname"]["disabled"] and p["ressave"]["disabled"]
    assert not p["resarm"]["hidden"]
    assert not [m for m in out["sent"] if "save" in m]
    assert [m for m in out["sent"] if m.get("kill")]


def test_no_maze_is_free_play_with_no_race_to_start():
    p = show(snap(race=...), race_maze=None)
    assert (p["timelabel"]["text"], p["timer"]["text"], p["timer"]["cls"]) == (
        "NOW PLAYING",
        "FREE PLAY",
        "free",
    )
    assert p["scores"]["hidden"] and p["gates"]["hidden"] and not p["nomaze"]["hidden"]
    assert p["racebtn"]["disabled"] and not p["arm"]["disabled"] and not p["kill"]["disabled"]


def test_stop_states_and_the_arm_block_reason_show_in_the_line_under_the_board():
    def line(*snaps):
        b = show(*snaps)["banner"]
        return b["text"], b["data"]["tone"]

    assert line(snap(board_lost=True)) == ("board lost, check the usb cable", "stop")
    assert line(snap(camera_lost=True)) == ("camera lost", "stop")
    assert line(snap(armed=True, killing=True)) == ("stopped", "stop")
    seen, gone = snap(armed=True, hand_seen=True), snap(armed=True, hand_seen=False)
    assert line(seen, gone) == ("HAND LOST", "stop")
    assert line(gone) != ("HAND LOST", "stop")
    assert line(snap(arm_blocked="exposure is 10 ms, want 3 ms")) == (
        "exposure is 10 ms, want 3 ms",
        "warn",
    )
    p = show(snap(armed=True, killing=True))
    assert (p["armed"]["text"], p["armed"]["cls"]) == ("KILLING", "chip warn")


def test_space_always_kills_and_never_clicks_a_focused_button():
    for on in (None, "arm", "racebtn", "kill"):
        steps = [
            OPEN,
            ("snap", snap()),
            ("down", ["Space", on]),
            ("up", ["Space", on]),
            ("frame", None),
        ]
        out = run(steps)
        assert out["out"][2:4] == [True, True], on  # both key events are swallowed
        assert out["sent"][-1].get("kill") is True and not any("arm" in m for m in out["sent"]), on
    out = run(
        [
            OPEN,
            ("snap", snap()),
            ("down", ["Enter", "arm"]),
            ("up", ["Enter", "arm"]),
            ("frame", None),
        ]
    )
    assert out["out"][2:4] == [True, True] and out["sent"][-1] == {"x": 0, "y": 0}


def test_buttons_and_keys_send_one_shot_commands():
    steps = [OPEN, ("snap", snap()), ("click", "arm"), ("down", ["KeyD", None]), ("frame", None)]
    steps += [
        ("snap", snap(armed=True)),
        ("down", ["KeyR", None]),
        ("up", ["KeyD", None]),
        ("frame", None),
    ]
    steps += [
        ("snap", snap(armed=True)),
        ("click", "kill"),
        ("down", ["KeyC", None]),
        ("frame", None),
    ]
    sent = run(steps)["sent"]
    assert sent[1:] == [
        {"x": 1, "y": 0, "arm": True},
        {"x": 0, "y": 0, "race": "start"},
        {"x": 0, "y": 0, "kill": True, "recentre": True},
    ]


def test_watch_tab_disables_arm_race_and_kill_and_sends_nothing():
    steps = [OPEN, ("snap", snap()), ("click", "arm"), ("click", "kill"), ("click", "racebtn")]
    steps += [("down", ["Space", None]), ("down", ["KeyD", None]), ("frame", None)]
    steps += [("snap", snap(armed=True)), ("frame", None)]
    out = run(steps, query="?watch")
    assert all(m == {} for m in out["sent"]) and len(out["sent"]) == 3
    assert all(out["page"][i]["disabled"] for i in ("arm", "kill", "racebtn"))


def picked(**over):
    return {"mode": "hand", "steer": "hand", "hand_mode": "tilt", "guard": None, **over}


def off(page, *groups):
    return [k["disabled"] for g in groups for k in page[g]["kids"]]


def test_choice_buttons_show_what_is_live_and_lock_while_armed():
    live = picked(mode="path", hand_mode="palm")
    p = show(snap(choices=live), config={"hand": True})
    assert pressed(p, "modes") == ["false", "false", "true"]
    assert pressed(p, "steers") == ["true", "false"] and pressed(p, "handmodes") == [
        "false",
        "true",
    ]
    assert not any(off(p, "modes", "steers", "handmodes")) and p["choices"]["cls"] == ""
    assert not p["handrow"]["hidden"]
    p = show(snap(armed=True, choices=live), config={"hand": True})
    assert all(off(p, "modes", "steers", "handmodes")) and p["guard"]["disabled"]
    assert p["choices"]["cls"] == "locked"
    assert pressed(show(snap(choices=picked(mode="assist"))), "modes") == ["false"] * 3


def test_hand_row_hides_on_keys_and_hand_needs_a_hand_source():
    p = show(snap(choices=picked(steer="keys")))
    assert p["handrow"]["hidden"] and off(p, "steers") == [True, False]
    assert pressed(p, "steers") == ["false", "true"]


def guard(page):
    g = page["guard"]
    return g["attrs"]["aria-checked"], g["disabled"], page["guardsay"]["text"]


def test_guard_switch_needs_a_build_with_a_guard_and_a_player_steering():
    assert guard(show(snap(choices=picked(guard=None)))) == ("false", True, "invisible walls")
    assert guard(show(snap(choices=picked(guard=False)))) == ("false", False, "invisible walls")
    assert guard(show(snap(choices=picked(guard=True)))) == ("true", False, "invisible walls")
    held = picked(mode="balance", guard=True)
    assert guard(show(snap(choices=held))) == ("false", True, "off in balance")


def test_clicked_choices_go_out_once_in_one_set_message():
    clicks = [("click", at) for at in ("modes/1", "steers/1", "handmodes/1", "guard")]
    steps = [OPEN, ("snap", snap(choices=picked(guard=False))), *clicks, ("frame", None)]
    out = run([*steps, ("snap", snap()), ("frame", None)], config={"hand": True})
    want = {"mode": "balance", "steer": "keys", "hand_mode": "palm", "guard": True}
    assert [m["set"] for m in out["sent"] if "set" in m] == [want]


def test_an_armed_page_and_a_watch_tab_send_no_choices():
    clicks = [("click", at) for at in ("modes/1", "steers/1", "handmodes/1", "guard")]
    live = picked(guard=False)
    for s, query in ((snap(armed=True, choices=live), ""), (snap(choices=live), "?watch")):
        out = run([OPEN, ("snap", s), *clicks, ("frame", None)], query=query, config={"hand": True})
        assert not [m for m in out["sent"] if "set" in m], query
        assert (
            all(off(out["page"], "modes", "steers", "handmodes"))
            and out["page"]["guard"]["disabled"]
        )


BALANCE = picked(mode="balance")
AT = {"src": "mouse", "mm": [60.0, -40.0], "found": True}


def test_balance_swaps_the_race_readouts_for_distance_speed_and_tilt():
    ball = {"x_mm": 0.0, "y_mm": 0.0, "vx_mm_s": 30.0, "vy_mm_s": -40.0, "valid": True, "t_s": 1}
    tilt = {"sent_x_deg": 0.3, "sent_y_deg": -0.4, "est_x_deg": 0.0, "est_y_deg": 0.0}
    p = show(snap(armed=True, choices=BALANCE, target=AT, ball=ball, plate_state=tilt))
    assert (p["timelabel"]["text"], p["timer"]["text"]) == ("DISTANCE FROM TARGET", "72.1 MM")
    assert (p["speed"]["text"], p["tiltnow"]["text"]) == ("50", "0.50°")
    assert p["scores"]["hidden"] and p["gates"]["hidden"] and p["nomaze"]["hidden"]
    assert (
        not p["balance"]["hidden"] and not p["targetfield"]["hidden"] and p["steerfield"]["hidden"]
    )
    assert pressed(p, "targets") == ["false", "true", "false"] and not any(off(p, "targets"))
    assert p["racebtn"]["disabled"] and p["race"]["style"]["cursor"] == "crosshair"
    assert p["sub"]["text"] == "BALANCE · CLICK THE BOARD TO MOVE THE TARGET"
    assert p["banner"]["text"] == "click the board or draw an x"
    assert p["handnone"]["text"] == "not used"
    gone = {**ball, "valid": False}
    assert show(snap(choices=BALANCE, target=AT, ball=gone))["timer"]["text"] == "--.- MM"


def test_play_mode_keeps_the_race_readouts_and_the_steering_row():
    p = show(snap())
    assert p["balance"]["hidden"] and p["targetfield"]["hidden"] and not p["steerfield"]["hidden"]
    assert p["race"]["style"]["cursor"] == "" and p["handnone"]["text"] == "no hand"


def targets(steps, **kw):
    return [m["target"] for m in run([OPEN, *steps], **kw)["sent"] if "target" in m]


def test_a_click_on_the_board_sends_the_spot_in_plate_mm_only_in_balance():
    held = ("snap", snap(armed=True, choices=BALANCE, target=AT))
    # the picture is the plate plus 17 mm of scale all round, +y up
    half = 305 / 2 + 17
    sent = targets([held, ("tap", ["race", 0.75, 0.25]), ("frame", None)])
    assert sent == [[pytest.approx(half / 2), pytest.approx(half / 2)]]
    assert targets([held, ("tap", ["race", 0.5, 1.0]), ("frame", None)]) == [[0, -half]]
    assert targets([("snap", snap()), ("tap", ["race", 0.75, 0.25]), ("frame", None)]) == []


def test_target_buttons_send_their_source_and_r_does_not_race_in_balance():
    held = ("snap", snap(armed=True, choices=BALANCE, target=AT))
    clicks = [
        ("click", "targets/0"),
        ("frame", None),
        held,
        ("click", "targets/2"),
        ("frame", None),
    ]
    out = run([OPEN, held, *clicks, held, ("down", ["KeyR", None]), ("frame", None)])
    assert [m["target"] for m in out["sent"] if "target" in m] == ["centre", "marker"]
    assert not [m for m in out["sent"] if "race" in m]


def test_the_page_keeps_asking_for_the_x_once_a_second_until_it_is_found():
    lost = ("snap", snap(choices=BALANCE, target={"src": "marker", "mm": [0, 0], "found": False}))
    found = ("snap", snap(choices=BALANCE, target={"src": "marker", "mm": [5, 5], "found": True}))
    step = [("frame", None), ("wait", 600)]
    assert targets([lost, *step, lost, *step, lost, *step, lost, *step]) == ["marker", "marker"]
    assert targets([found, *step, ("wait", 2000), found, *step]) == []
    assert targets([lost, *step, ("wait", 2000), lost, *step], query="?watch") == []
    p = show(lost[1])
    assert (p["banner"]["text"], p["banner"]["data"]["tone"]) == ("click arm to start", "")
    p = show(snap(armed=True, choices=BALANCE, target=lost[1]["target"]))
    assert p["banner"]["text"] == "no x on the board yet, draw one"


def test_settings_link_goes_dead_while_armed_so_the_kill_key_stays_in_reach():
    assert show(snap())["settings"]["cls"] == "pill"
    assert show(snap(armed=True))["settings"]["cls"] == "pill off"


V, T = ("down", ["KeyV", None]), ("down", ["KeyT", None])


def looks(steps, **kw):
    out = run([OPEN, *steps], **kw)
    p = out["page"]
    return p["viewtag"]["text"], p["race"]["cls"], p["minimap"]["hidden"], out["sent"]


def test_v_steps_through_the_views_t_is_top_down_at_once_and_nothing_is_sent():
    s = ("snap", snap())
    plain = looks([s])
    assert plain[:3] == ("TOP VIEW", "", True)
    assert looks([V, s])[:3] == ("FOLLOW CAM 3X", "follow", False)
    assert looks([V, V, s])[:3] == ("CAM 2 WIDE", "", True)
    assert looks([V, V, V, s])[:3] == plain[:3]
    assert looks([V, T, s])[:3] == plain[:3] and looks([V, V, T, s])[:3] == plain[:3]
    assert looks([("click", "viewtag"), s])[:3] == ("FOLLOW CAM 3X", "follow", False)
    assert looks([V, V, s, ("frame", None)])[3] == plain[3] + [{"x": 0, "y": 0}]


def test_view_keys_work_on_a_watch_tab_and_stay_out_of_the_name_field():
    s = ("snap", snap())
    assert looks([V, s], query="?watch")[:3] == ("FOLLOW CAM 3X", "follow", False)
    done = ("snap", snap(race=DONE))
    typing = [done, ("wait", 1000), done, ("down", ["KeyV", "resname"]), done]
    assert looks(typing)[0] == "TOP VIEW"


def test_the_3d_tag_names_the_shot_the_camera_is_on():
    def racing(t, state="running", **kw):
        ball = {"x_mm": 85.0 - 20 * t, "y_mm": 85.0, "vx_mm_s": -20.0, "vy_mm_s": 0.0}
        live = race(state, elapsed_s=t, reached=0, **kw)
        return (
            "snap",
            snap(t_s=50 + t, armed=True, ball={**ball, "valid": True, "t_s": t}, race=live),
        )

    early = [racing(i / 30) for i in range(30)]
    later = [racing(i / 30) for i in range(30, 75)]
    assert looks([V, V, *early])[0] == "CAM 2 WIDE"
    assert looks([V, V, *early, *later])[0] == "CAM 3 CHASE"
    done = [racing(2.5, "finished", score={"time_s": 2.5, "penalty_s": 0.0, "checkpoints": 4})]
    assert looks([V, V, *early, *later, *done])[0] == "CAM 1 TOWER"
    # no maze, no path to follow: it stays wide
    free = [("snap", snap(t_s=50 + i / 30, armed=True, race=...)) for i in range(90)]
    assert looks([V, V, *free], race_maze=None)[0] == "CAM 2 WIDE"


def test_a_board_click_only_moves_the_target_in_the_flat_view():
    held = ("snap", snap(armed=True, choices=BALANCE, target=AT))
    tap = [("tap", ["race", 0.75, 0.25]), ("frame", None)]
    assert len(targets([held, *tap])) == 1
    assert targets([V, held, *tap]) == [] and targets([V, V, held, *tap]) == []
    assert run([OPEN, V, held])["page"]["race"]["style"]["cursor"] == ""


def test_a_snapshot_the_page_cannot_draw_never_holds_back_the_kill_key():
    broken = snap(armed=True, choices=...)
    steps = [OPEN, ("snap", snap(armed=True)), ("frame", None), ("snap", broken)]
    out = run(
        [*steps, ("down", ["Space", None]), ("frame", None), ("snap", broken), ("frame", None)]
    )
    assert "TypeError" in out["out"][3] and out["out"][1] is None
    assert [m for m in out["sent"] if m.get("kill")] == [{"x": 0, "y": 0, "kill": True}]
    assert len(out["sent"]) == 4
