import math
import random
import shutil

import pytest

from tests.unit.cam import DT, frame, play, shots, steps

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("node") is None, reason="needs node"),
]

# east along the bottom, then north up the right side
PATH = [[-100, -100], [100, -100], [100, 100]]
CORNER, END = [100, -100], [100, 100]
HOLD_S = 1.0


def times(seconds, start=0.0):
    return [start + i * DT for i in range(round(seconds / DT))]


def east(t, speed=30.0):
    return -100 + speed * t, -100.0


def test_idle_is_wide_a_race_opens_wide_then_chases_and_the_finish_cuts_straight_down():
    idle = [frame(t, -100, -100, phase="idle", path=PATH) for t in times(1.0)]
    run = [frame(1 + t, *east(t), run_s=t, gate=END, path=PATH) for t in times(1.5)]
    done = [frame(2.5 + t, *east(1.5), phase="done", path=PATH) for t in times(0.5)]
    got = shots(play(idle + run + done))
    assert [s for s, _, _ in got] == ["wide", "chase", "tower"]
    assert got[1][1] == pytest.approx(1 + 1.2, abs=DT)
    # the chase had only been up 0.3 s, the finish does not wait for the hold
    assert got[2][1] == pytest.approx(2.5, abs=DT)


def round_the_corner(t, speed=30.0):
    d = speed * t
    return (-100 + d, -100.0) if d < 200 else (100.0, -100 + (d - 200))


def test_gate_cam_runs_from_just_before_a_gate_is_taken_until_the_ball_is_clear_of_it():
    ring = 30
    frames = []
    for t in times(11.0):
        x, y = round_the_corner(t)
        taken = math.dist((x, y), CORNER) < ring or y > -100
        frames.append(frame(t, x, y, gate=END if taken else CORNER, path=PATH, gate_r=ring))
    got = play(frames)
    cuts = shots(got)
    assert [s for s, _, _ in cuts] == ["wide", "chase", "gate", "chase"]
    # in: 40 mm before the ring takes the gate. out: 40 mm past the gate itself
    gate_on = next(s for s in got if s["shot"] == "gate")
    assert math.dist(gate_on["eye"], CORNER) == pytest.approx(ring + 40, abs=1.5)
    assert all(s["filmed"] == CORNER for s in got if s["shot"] == "gate")
    back = next(s for s in got if s["t"] > cuts[2][2])
    assert math.dist(back["eye"], CORNER) == pytest.approx(40, abs=1.5)
    # the gate cam looks between the ball and the gate, tighter and lower than the chase
    held = [s for s in got if s["shot"] == "gate"][-1]["cam"]
    assert held["zoom"] > 2.2 and held["pitch"] > 68
    assert math.dist((held["cx"], held["cy"]), CORNER) < 25


def test_gates_with_no_ring_are_filmed_from_40_mm_out():
    frames = [frame(t, *east(t), gate=CORNER, path=PATH) for t in times(6.5)]
    got = play(frames)
    gate_on = next(s for s in got if s["shot"] == "gate")
    assert math.dist(gate_on["eye"], CORNER) == pytest.approx(40, abs=1.5)


def test_every_shot_is_held_a_second_however_the_ball_dithers_on_the_gate_line():
    frames = []
    for i, t in enumerate(times(8.0)):
        # 35 or 45 mm from the gate, swapping every tenth of a second
        x = 100 - (35 if (i // 6) % 2 else 45)
        frames.append(frame(t, x, -100, gate=CORNER, path=PATH))
    got = shots(play(frames))
    assert len(got) >= 5 and {s for s, _, _ in got[1:]} == {"chase", "gate"}
    assert all(end - start >= HOLD_S - 2 * DT for _, start, end in got[:-1])


def test_with_no_path_the_camera_stays_wide_on_the_whole_board():
    frames = [frame(t, *east(t), gate=CORNER, path=None) for t in times(4.0)]
    got = play(frames)
    assert {s["shot"] for s in got} == {"wide"}
    assert all(abs(s["cam"]["cx"]) < 1e-9 and abs(s["cam"]["cy"]) < 1e-9 for s in got)
    assert max(abs(s["cam"]["yaw"]) for s in got) <= 12


def test_a_noisy_ball_does_not_shake_the_camera():
    rnd = random.Random(3)
    clean, noisy = [], []
    for t in times(5.0):
        x, y = east(t, speed=25.0)
        clean.append(frame(t, x, y, gate=END, path=PATH))
        jit = (x + rnd.uniform(-2, 2), y + rnd.uniform(-2, 2))
        noisy.append(frame(t, *jit, gate=END, path=PATH))
    a, b = play(clean), play(noisy)
    chase = [i for i, s in enumerate(b) if s["shot"] == "chase" and s["t"] > 2.5]
    assert len(chase) > 100
    # 2 mm of jitter in, a small fraction of that out, frame to frame and overall
    assert max(steps(b, "cx")[i - 1] for i in chase) < 0.7
    assert max(steps(b, "cy")[i - 1] for i in chase) < 0.2
    assert max(steps(b, "yaw")[i - 1] for i in chase) < 0.05
    assert max(abs(b[i]["cam"]["cy"] - a[i]["cam"]["cy"]) for i in chase) < 0.6


def test_a_briefly_lost_ball_holds_the_picture_and_a_long_loss_falls_back_to_wide():
    def ball(t):
        # the tracker reports 0, 0 with valid false while it has no ball
        lost = 3.0 <= t < 3.3 or t >= 5.0
        return (0.0, 0.0, False) if lost else (*east(t, 25.0), True)

    got = play(frame(t, *ball(t), gate=END, path=PATH) for t in times(7.0))
    by_t = {round(s["t"], 4): s for s in got}
    blip = [s for s in got if 3.0 <= s["t"] < 3.3]
    assert {s["shot"] for s in blip} == {"chase"}
    # nowhere near the 0, 0 it was handed, and no jolt on the way out or back in
    eye_before = by_t[round(times(3.0)[-1], 4)]["eye"]
    assert all(s["eye"] == eye_before for s in blip)
    around = [i for i, s in enumerate(got) if 2.9 <= s["t"] <= 3.6]
    assert max(steps(got, "cx")[i] for i in around) < 0.7
    assert max(steps(got, "cy")[i] for i in around) < 0.05
    names = [(s, round(start, 1)) for s, start, _ in shots(got)]
    assert names == [("wide", 0.0), ("chase", 1.2), ("wide", 6.0)]
    # the fall back to wide is an ease, not a jump
    after = [i for i, s in enumerate(got) if s["t"] >= 6.0]
    assert max(steps(got, "cx")[i - 1] for i in after) < 8


def test_a_tracker_jump_for_a_couple_of_frames_does_not_move_the_camera():
    frames = [frame(t, 0.0, -100.0, gate=END, path=PATH) for t in times(6.0)]
    still = play(frames)
    for i in (240, 241):
        frames[i] = frame(frames[i]["t"], 100.0, 0.0, gate=END, path=PATH)
    got = play(frames)
    assert [s["cam"] for s in got] == [s["cam"] for s in still]
    assert {s["shot"] for s in got[240:]} == {"chase"}


def test_a_tracker_hopping_between_far_spots_is_never_believed_and_ends_up_wide():
    spots = [(0.0, -100.0), (90.0, 60.0), (-80.0, 20.0), (60.0, -20.0)]

    def at(i, t):
        return spots[0] if t < 3.0 else spots[1 + i % 3]

    got = play(frame(t, *at(i, t), gate=END, path=PATH) for i, t in enumerate(times(6.0)))
    hopping = [s for s in got if s["t"] >= 3.0]
    assert all(s["eye"] == hopping[0]["eye"] for s in hopping)
    assert [(n, round(start, 1)) for n, start, _ in shots(got)] == [
        ("wide", 0.0),
        ("chase", 1.2),
        ("wide", 4.0),
    ]


def test_a_ball_that_really_is_somewhere_else_is_taken_after_a_moment_and_eased_to():
    def at(t):
        return (0.0, -100.0) if t < 4.0 else (80.0, -100.0)

    got = play(frame(t, *at(t), gate=END, path=PATH) for t in times(8.0))
    held = [s for s in got if 4.0 <= s["t"] < 4.1]
    assert all(s["eye"] == held[0]["eye"] for s in held)
    assert got[-1]["eye"][0] == pytest.approx(80, abs=0.5)
    late = [i for i, s in enumerate(got) if s["t"] > 3.0]
    assert max(steps(got, "cx")[i - 1] for i in late) < 3
    assert max(steps(got, "yaw")[i - 1] for i in late) < 0.05
    assert {s["shot"] for s in got if s["t"] > 1.3} == {"chase"}


def turn(states, t0, t1):
    return [s["cam"]["yaw"] for s in states if t0 <= s["t"] <= t1]


def test_heading_follows_the_path_and_turns_the_short_way_without_a_snap():
    # east, a short hop north, then back west: the plate ends up turned half way round
    u_turn = [[-100, 0], [100, 0], [100, 20], [-100, 20]]

    def at(t):
        d = 40.0 * max(0.0, t - 1.5)
        if d < 150:
            return -50 + d, 0.0
        if d < 170:
            return 100.0, d - 150
        return 100 - (d - 170), 20.0

    got = play(frame(t, *at(t), gate=None, path=u_turn) for t in times(12.0))
    assert {s["shot"] for s in got if s["t"] > 1.3} == {"chase"}
    # heading east: yaw 90 puts +x up the screen
    assert turn(got, 4.0, 4.5)[-1] == pytest.approx(90, abs=1)
    # heading west at the end: it went on round through north, never back the long way
    yaws = turn(got, 1.3, 12.0)
    assert yaws[-1] == pytest.approx(270, abs=2) or yaws[-1] == pytest.approx(-90, abs=2)
    assert max(abs(b - a) for a, b in zip(yaws, yaws[1:], strict=False)) < 4


def test_a_clock_that_goes_back_starts_the_camera_over():
    before = [frame(500 + t, *east(t), run_s=t, gate=END, path=PATH) for t in times(3.0)]
    after = [frame(t, *east(t), gate=END, path=PATH) for t in times(3.0)]
    got = play(before + after)
    fresh = play(after)
    assert [s["shot"] for s in got[len(before) :]] == [s["shot"] for s in fresh]
    assert got[-1]["shot"] == "chase" and got[len(before)]["cam"] == fresh[0]["cam"]
