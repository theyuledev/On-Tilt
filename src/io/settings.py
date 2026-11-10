import json
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "settings.toml"
LOCAL = ROOT / "configs" / "settings.local.toml"


def _values(section: dict) -> dict:
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in section.items()
    }


def load_raw(base: Path = BASE, local: Path | None = None) -> dict:
    local = LOCAL if local is None else local
    with base.open("rb") as f:
        raw = tomllib.load(f)
    if local.exists():
        with local.open("rb") as f:
            for name, section in tomllib.load(f).items():
                raw.setdefault(name, {}).update(section)
    return raw


def load(base: Path = BASE, local: Path | None = None) -> dict:
    return {name: _values(section) for name, section in load_raw(base, local).items()}


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list | tuple):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {toml_value(x)}" for k, x in v.items()) + " }"
    return repr(v)


def _put(lines: list[str], section: str, key: str, line: str) -> list[str]:
    """lines with `line` in place of the section's key, or added at the section's end."""
    head = next((i for i, text in enumerate(lines) if text.strip() == f"[{section}]"), None)
    if head is None:
        return [*lines, *([""] if lines else []), f"[{section}]", line]
    end = next((i for i in range(head + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    for i in range(head + 1, end):
        if re.match(rf"{re.escape(key)}\s*=", lines[i]):
            return [*lines[:i], line, *lines[i + 1 :]]
    while not lines[end - 1].strip():
        end -= 1
    return [*lines[:end], line, *lines[end:]]


def save_local(changes: dict[tuple[str, str], object], label: str, local: Path | None = None):
    """Writes changed values into the local file and leaves its other lines alone. A number
    carries a label in the base file, so it gets `label` here; the base file is never written."""
    local = LOCAL if local is None else local
    with BASE.open("rb") as f:
        base = tomllib.load(f)
    text = local.read_text() if local.exists() else ""
    before, lines = tomllib.loads(text), text.splitlines()
    for (section, key), v in changes.items():
        if isinstance(base.get(section, {}).get(key), dict):
            v = {"value": v, "label": label}
        before.setdefault(section, {})[key] = v
        lines = _put(lines, section, key, f"{key} = {toml_value(v)}")
    text = "\n".join(lines) + "\n"
    # a value spread over several lines can't be swapped line by line, so check before writing
    if tomllib.loads(text) != before:
        raise ValueError(f"{local.name} needs a hand edit, nothing was saved")
    tmp = local.with_name(local.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, local)


def env(key: str, default: str | None = None) -> str | None:
    if key in os.environ:
        return os.environ[key]
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip()
    return default
