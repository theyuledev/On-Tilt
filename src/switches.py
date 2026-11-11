"""The play choices the page can switch: mode, what steers, how the hand reads, the edge guard."""

from src.contracts import Choices

ALLOWED = {
    "mode": ("hand", "balance", "path"),
    "steer": ("hand", "keys"),
    "hand_mode": ("tilt", "palm"),
}


def parse(raw: dict) -> Choices:
    """Choices from a page message; a value the page may not pick is a ValueError."""
    picked = {k: raw[k] for k in (*ALLOWED, "guard") if raw.get(k) is not None}
    for k, v in picked.items():
        if not (isinstance(v, bool) if k == "guard" else v in ALLOWED[k]):
            raise ValueError(f"{k} can't be {v!r}")
    return Choices(**picked)


def view(loop) -> Choices:
    control, hand = loop.cfg["control"], loop.hand
    return Choices(
        mode=control["mode"],
        steer="keys" if hand is None or loop.keys_only else "hand",
        hand_mode=(hand.mode if hand is not None else None) or loop.cfg["input"]["hand_mode"],
        # None on a build without the edge guard
        guard=control.get("guard_on"),
    )


def choose(loop, c: Choices) -> None:
    """Takes the page's choices on a disarmed loop; an armed one keeps what it has."""
    if loop.safety.armed:
        return
    control = loop.cfg["control"]
    if c.mode is not None:
        loop.mode = c.mode  # raises before anything is kept if the mode can't be built
        control["mode"] = c.mode
    if c.steer is not None:
        loop.keys_only = c.steer == "keys"
    if c.hand_mode is not None and loop.hand is not None:
        loop.hand.set_mode(c.hand_mode)
    if c.guard is not None and "guard_on" in control:
        control["guard_on"] = c.guard
