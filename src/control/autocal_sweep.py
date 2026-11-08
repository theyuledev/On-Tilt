import numpy as np

from src.control.autocal_fit import probe_fit
from src.sim.plate import SimActuator


class DeadSweep:
    """A servo model per candidate dead time, fed the probe's pulses. The one whose probe fit
    misfits least is the best guess of the real chain, printed next to the stored one."""

    def __init__(self, rig: dict, c: dict):
        lo, hi, step = c["autocal_dead_sweep_s"]
        dead = np.round(np.arange(lo, hi + step / 2, step), 3)
        self.models = {float(d): SimActuator(rig, {**c, "servo_dead_time_s": d}) for d in dead}
        self.mod: dict = {d: [] for d in self.models}

    def send(self, cmd, now: float, level) -> None:
        for d, m in self.models.items():
            m.send(cmd, now)
            self.mod[d].append((now, *(np.array(m.pos) - level)))

    def best(self, rows, t0: float) -> float:
        t, u, v = (np.array(a) for a in zip(*rows, strict=True))
        rms = {}
        for d, mod in self.mod.items():
            tm, ox, oy = (np.array(a) for a in zip(*mod, strict=True))
            rms[d] = probe_fit(t, np.c_[u, v], tm, np.c_[ox, oy], t0)[2]
        return min(rms, key=rms.get)
