import math

from src.contracts import ServoCommand
from src.control.autocal_fit import second_amp, solve
from src.control.pulse_map import center
from src.sim.fit import accel_vec

ORDER = (("x", 1), ("x", -1), ("y", 1), ("y", -1))


class Autocal:
    """Pulse schedule for self calibration. Tick it with (now, observation); it hands back the
    pulses to hold, then None once done or aborted. Positions are flat plate mm, y up."""

    def __init__(self, cfg: dict, centre_px: tuple[float, float], s_ball: float):
        self.rig, self.c = cfg["rig"], cfg["control"]
        self.centre, self.s = centre_px, s_ball
        self.level = center(self.rig)
        self.wall = (self.rig["plate_inner_mm"] - self.rig["ball_d_mm"]) / 2
        self.wall -= self.c["autocal_rim_margin_mm"]
        self.amps = [self.c["autocal_a1_us"], self.c["autocal_a2_max_us"]]
        self.queue = [(ax, s, 0) for ax, s in ORDER]
        self.acc: dict = {}
        self.rows: list[tuple[float, float, float]] = []
        self.t_start: float | None = None
        self.t_seen: float | None = None
        self.aborted: str | None = None
        self.result = None

    @property
    def running(self) -> bool:
        return self.result is None and self.aborted is None

    def step(self, now: float, obs) -> ServoCommand | None:
        if self.running:
            self._watch(now, obs)
        if self.t_start is None:
            self.t_start = now
        cyc = 2 * self.c["autocal_pulse_s"] + self.c["autocal_settle_s"]
        if self.running and now - self.t_start >= cyc:
            self._fit()
            self.t_start = now
        if not self.running:
            return None
        ax, s, i = self.queue[0]
        el, pulse = now - self.t_start, self.c["autocal_pulse_s"]
        off = s * self.amps[i] if el < pulse else -s * self.amps[i] if el < 2 * pulse else 0
        cx, cy = self.level
        if ax == "x":
            return ServoCommand(round(cx + off), round(cy))
        return ServoCommand(round(cx), round(cy + off))

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

    def _fit(self) -> None:
        ax, s, i = self.queue.pop(0)
        amp, c = self.amps[i], self.c
        t0 = self.t_start + c["servo_dead_time_s"] + amp / c["servo_top_speed_us_s"]
        t0 += 3 * c["servo_tau_s"]
        t1 = self.t_start + c["autocal_pulse_s"] + c["servo_dead_time_s"]
        t, u, v = zip(*[r for r in self.rows if t0 <= r[0] <= t1], strict=True)
        a = accel_vec(t, u, v, t0, t1)
        self.acc[(ax, s, i)] = a
        if (ax, s, i) == ("x", 1, 0):
            self.amps[1] = second_amp(
                amp, math.hypot(*a), c["autocal_pulse_s"], c["autocal_travel_mm"], self.amps[1]
            )
        if not self.queue and i == 0:
            self.queue = [(ax, s, 1) for ax, s in ORDER]
        elif not self.queue:
            try:
                self.result = solve(
                    self.acc,
                    tuple(self.amps),
                    (self.rig["trim_x_us"], self.rig["trim_y_us"]),
                    c["servo_dead_band_us"],
                    c["sim_g_mm_s2"],
                    c["autocal_mirror_tol_deg"],
                )
            except ValueError as e:
                self.aborted = str(e)
