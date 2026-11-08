import math
from bisect import bisect_left, bisect_right

import numpy as np

from src.contracts import ServoCommand
from src.control.autocal_fit import (
    Calibration,
    agree,
    doublet_fit,
    mapping,
    per_deg,
    probe_fit,
    second_amp,
    stored_m,
    trim_step,
)
from src.control.autocal_hold import Hold
from src.control.pulse_map import center
from src.sim.plate import SimActuator

ORDER = (("x", 1), ("x", -1), ("y", 1), ("y", -1))
PROBE = ((1, 1), (-1, -1), (-1, 1), (1, -1))  # x, y sign per slot


class Autocal:
    """Probe the plate both axes at once, then a held doublet per axis and sign. Tick it with
    (now, observation); it hands back the pulses, then None once done or aborted. Positions are
    flat plate mm, y up. quick: the probe is checked against the stored mapping and only the
    trims are new, unless they disagree, then it runs the full set."""

    def __init__(self, cfg: dict, centre_px: tuple[float, float], s_ball: float, quick=False):
        self.rig, self.c, self.vision = cfg["rig"], cfg["control"], cfg["vision"]
        self.centre, self.s = centre_px, s_ball
        self.c0 = np.array(center(self.rig), dtype=float)
        self.level = self.c0.copy()
        self.wall = (self.rig["plate_inner_mm"] - self.rig["ball_d_mm"]) / 2
        self.wall -= self.c["autocal_rim_margin_mm"]
        self.model = SimActuator(self.rig, self.c)  # the stored servo chain, fed what we send
        self.rows: list[tuple[float, float, float]] = []
        self.mod: list[tuple[float, float, float]] = []
        self.m: dict = {}
        self.b: dict = {}
        self.rms: list[float] = []
        self.amps = [self.c["autocal_a1_us"], self.c["autocal_a2_max_us"]]
        self.quick = quick
        sets = (1,) if quick else (0, 1)
        self.queue = [("p", 0, 0)] + [(ax, s, i) for i in sets for ax, s in ORDER]
        self.M = None
        self.hold = False
        self.holder = Hold(self.c, self.vision["velocity_span_frames"])
        self.t0 = self.t_seen = None
        self.note = ""
        self.aborted: str | None = None
        self.result = None

    @property
    def running(self) -> bool:
        return self.result is None and self.aborted is None

    def step(self, now: float, obs) -> ServoCommand | None:
        if self.running:
            self._watch(now, obs)
        if not self.running:
            return None
        if self.t0 is None:
            self.t0 = now
        d = self._hold(now) if self.hold else self._pulse(now)
        if d is None or not self.running:
            return None
        r = self.rig["startup_range_us"]
        x, y = np.clip(self.level + d, self.c0 - r, self.c0 + r)
        cmd = ServoCommand(round(float(x)), round(float(y)))
        self.model.send(cmd, now)
        self.mod.append((now, *(np.array(self.model.pos) - self.level)))
        return cmd

    def _watch(self, now: float, obs) -> None:
        if self.t_seen is None:
            self.t_seen = now
        if obs is not None and obs.ball_px.valid:
            self.t_seen = now
            if not self.rows or obs.t_read > self.rows[-1][0]:
                u = (obs.ball_px.x_px - self.centre[0]) / self.s
                v = (self.centre[1] - obs.ball_px.y_px) / self.s
                self.rows.append((obs.t_read, u, v))
                if math.hypot(u, v) > self.wall:
                    self.aborted = f"ball at {math.hypot(u, v):.0f} mm, too near the rim"
        if now - self.t_seen > self.c["autocal_lost_s"]:
            self.aborted = "lost the ball"

    def _pulse(self, now: float):
        kind, s, i = self.queue[0]
        el, c = now - self.t0, self.c
        if kind == "p":
            slot = c["autocal_probe_slot_s"]
            if el >= 4 * slot + c["autocal_tail_s"]:
                return self._next(now)
            k = int(el // slot)
            return c["autocal_probe_us"] * np.array(PROBE[k]) if k < 4 else np.zeros(2)
        T = c["autocal_pulse_s"]
        if el >= 2 * T + c["autocal_tail_s"]:
            return self._next(now)
        off = s * self.amps[i] if el < T else -s * self.amps[i] if el < 2 * T else 0.0
        return np.array([off, 0.0]) if kind == "x" else np.array([0.0, off])

    def _next(self, now: float):
        job = self.queue.pop(0)
        if job[0] == "p":
            self._probe(now)
        else:
            self._doublet(job, now)
        self.t0, self.hold = now, True
        self.holder.start(now)
        return self._hold(now) if self.running else None

    def _hold(self, now: float):
        if not self.queue:
            self._finish()
            return None
        u, why = self.holder.step(now, self.rows, self.M)
        if why == "still":
            self.hold, self.t0 = False, now
            return self._pulse(now)
        if why:
            self.aborted = why
        return u

    def _window(self, t0: float, t1: float, axis: int | None):
        key = lambda r: r[0]  # noqa: E731
        win = self.rows[bisect_left(self.rows, t0, key=key) : bisect_right(self.rows, t1, key=key)]
        mod = self.mod[bisect_left(self.mod, t0, key=key) : bisect_right(self.mod, t1, key=key)]
        if len(win) < 2 * (5 if axis is None else 4) or len(mod) < 2:  # twice the unknowns
            self.aborted = f"only {len(win)} frames in a pulse"
            return None
        t, u, v = (np.array(a) for a in zip(*win, strict=True))
        tm, ox, oy = (np.array(a) for a in zip(*mod, strict=True))
        if axis is None:
            return probe_fit(t, np.c_[u, v], tm, np.c_[ox, oy], t0)
        return doublet_fit(t, np.c_[u, v], tm, np.c_[ox, oy], t0, axis)

    def _probe(self, now: float) -> None:
        c = self.c
        fit = self._window(self.t0, now, None)
        if fit is None:
            return
        m, b, rms = fit
        col = np.hypot(m[0], m[1])
        gain = col * np.abs([self.rig["us_per_deg_x"], self.rig["us_per_deg_y"]])
        lo, hi = (g * per_deg(c["sim_g_mm_s2"]) for g in c["autocal_gain_band"])
        flat = abs(np.linalg.det(m)) < c["autocal_parallel_sin"] * col[0] * col[1]
        if flat or rms > c["autocal_probe_rms_mm"] or not all(lo <= g <= hi for g in gain):
            self.aborted = "plate did not answer the probe"
            return
        self.M = m
        ref = stored_m(self.rig, self.vision["plate_yaw_deg"], c["sim_g_mm_s2"])
        if self.quick and not agree(m, ref, c["autocal_agree_deg"], c["autocal_agree_gain"]):
            self.queue = [(ax, s, i) for i in (0, 1) for ax, s in ORDER]
            self.note = "recalibrating"
        a1 = self.amps[0]
        self.amps[1] = second_amp(
            a1, np.hypot(*m[:, 0]) * a1, c["autocal_pulse_s"], c["autocal_travel_mm"], self.amps[1]
        )
        lim = c["autocal_level_max_us"]
        self.level = self.c0 + np.clip(-c["autocal_ff_gain"] * np.linalg.solve(m, b), -lim, lim)

    def _doublet(self, job, now: float) -> None:
        kind, s, i = job
        j, c = "xy".index(kind), self.c
        end = self.t0 + c["autocal_fit_end"] * c["autocal_pulse_s"]
        fit = self._window(self.t0, min(now, end), j)
        if fit is None:
            return
        self.m[job], self.b[job], rms = fit
        self.rms.append(rms)
        if s == -1:
            # the +- pair cancels rolling friction on this axis
            self.M[:, j] = (self.m[(kind, 1, i)] + self.m[job]) / 2
            step = trim_step(self.M, (self.b[(kind, 1, i)] + self.b[job]) / 2, j)
            lim = c["autocal_level_max_us"]
            self.level[j] = np.clip(self.level[j] + step, self.c0[j] - lim, self.c0[j] + lim)

    def _finish(self) -> None:
        g, rig = self.c["sim_g_mm_s2"], self.rig
        try:
            yaw, mirror, ux, uy = mapping(self.M, g, self.c["autocal_mirror_tol_deg"])
        except ValueError as e:
            self.aborted = str(e)
            return
        if self.quick and not self.note:
            # the quick gain reads too noisy to keep, only the trims are new
            yaw, ux, uy = self.vision["plate_yaw_deg"], rig["us_per_deg_x"], rig["us_per_deg_y"]
            mirror = uy < 0
        tx, ty = self.level - (rig["center_x_us"], rig["center_y_us"])
        resid = float(np.sqrt(np.mean(np.square(self.rms))))
        self.result = Calibration(float(tx), float(ty), ux, uy, yaw, mirror, resid)
