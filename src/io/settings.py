import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs" / "settings.toml"
LOCAL = ROOT / "configs" / "settings.local.toml"


def _values(section: dict) -> dict:
    return {
        k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in section.items()
    }


def load_raw(base: Path = BASE, local: Path = LOCAL) -> dict:
    with base.open("rb") as f:
        raw = tomllib.load(f)
    if local.exists():
        with local.open("rb") as f:
            for name, section in tomllib.load(f).items():
                raw.setdefault(name, {}).update(section)
    return raw


def load(base: Path = BASE, local: Path = LOCAL) -> dict:
    return {name: _values(section) for name, section in load_raw(base, local).items()}


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
