import pytest

from src.io import settings


@pytest.fixture(autouse=True)
def no_local_settings(monkeypatch, tmp_path):
    # tests see the checked in numbers, not this mac's settings.local.toml
    missing = tmp_path / "settings.local.toml"
    monkeypatch.setattr(settings, "LOCAL", missing)
    for fn in (settings.load, settings.load_raw):
        monkeypatch.setattr(fn, "__defaults__", (settings.BASE, missing))
