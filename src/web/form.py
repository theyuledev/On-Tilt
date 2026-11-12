"""The settings page's rules: what it shows, what it may change and what a change must be."""

import math
from dataclasses import asdict

from src import switches
from src.contracts import Choices
from src.games import maze
from src.io import settings
from src.modes import MODES, make_mode
from src.modes.spot import held

# servo limits, the pulse map, the safety times and the board link stay with the settings file
LOCKED = ("rig", "bench")
HIDDEN = ("bench",)  # what the tracker is scored against, not the game's to show
LABEL = "player-tuned (settings page)"
# divided by or used as a size while the app runs
POSITIVE = (
    ("vision", "px_per_mm"),
    ("vision", "line_stroke_mm"),
    ("vision", "line_min_mm"),
    ("vision", "line_step_mm"),
    ("control", "sim_tick_hz"),
)
ONE_OF = {
    ("control", "mode"): tuple(MODES),
    ("input", "hand_mode"): switches.ALLOWED["hand_mode"],
}


def view(loop) -> dict:
    """The running value of every setting in the files, by section, and what locks them."""
    saved, cfg = settings.load(), loop.cfg
    sections = {s: {k: cfg[s].get(k, v) for k, v in sec.items()} for s, sec in saved.items()}
    sections = {s: sec for s, sec in sections.items() if s not in HIDDEN}
    return {
        "armed": loop.safety.armed,
        "locked": list(LOCKED),
        "sections": sections,
        # json drops the point of a whole float, the page writes these with it
        "floats": [
            f"{s}.{k}" for s, sec in sections.items() for k, v in sec.items() if type(v) is float
        ],
        "mazes": sorted(p.stem for p in maze.MAZES.glob("*.toml")),
        "choices": asdict(switches.view(loop)),
        "target": held(loop.cfg)["src"],
    }


def coerce(old, new):
    """new as the kind of value old is; anything else is a ValueError."""
    if isinstance(old, bool) or isinstance(old, str):
        if type(new) is not type(old):
            raise ValueError(f"wants {type(old).__name__}")
        return new
    if isinstance(old, list):
        if not isinstance(new, list) or len(new) != len(old):
            raise ValueError(f"wants a list of {len(old)}")
        return [coerce(o, n) for o, n in zip(old, new, strict=True)]
    if isinstance(new, bool) or not isinstance(new, int | float) or not math.isfinite(new):
        raise ValueError("wants a number")
    if isinstance(old, int) and new != int(new):
        raise ValueError("wants a whole number")
    return type(old)(new)


def check(cfg: dict, changes: dict) -> dict[tuple[str, str], object]:
    """The changes by (section, key), each as its setting's kind; the first bad one raises."""
    saved, out = settings.load(), {}
    for path, new in changes.items():
        section, _, key = str(path).partition(".")
        if key not in saved.get(section, {}):
            raise ValueError(f"{path} is not a setting")
        if section in LOCKED:
            raise ValueError(f"{path} is read only here, change it in the settings file")
        try:
            value = coerce(cfg[section][key], new)
        except ValueError as e:
            raise ValueError(f"{path} {e}") from None
        if (section, key) in POSITIVE and value <= 0:
            raise ValueError(f"{path} must be above 0")
        if value not in ONE_OF.get((section, key), (value,)):
            raise ValueError(f"{path} must be one of {', '.join(ONE_OF[section, key])}")
        out[section, key] = value
    return out


def save(loop, changes: dict) -> None:
    """Checks the page's changes, writes them to the local settings file and runs on them.
    Nothing is kept unless the maze loads and the mode builds with the new values."""
    values = check(loop.cfg, changes)
    new = {s: dict(sec) for s, sec in loop.cfg.items()}
    for (section, key), value in values.items():
        new[section][key] = value
    if ("game", "maze") in values:
        try:
            maze.load(new["game"]["maze"])
        except (OSError, KeyError, TypeError) as e:
            raise ValueError(f"game.maze does not load: {e!r}") from None
    make_mode(new["control"]["mode"], new)
    settings.save_local(values, LABEL)
    for (section, key), value in values.items():
        loop.cfg[section][key] = value
    picked = Choices(mode=new["control"]["mode"], hand_mode=values.get(("input", "hand_mode")))
    switches.choose(loop, picked)
