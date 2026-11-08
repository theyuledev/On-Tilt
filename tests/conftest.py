import pytest

from src.io import settings


@pytest.fixture(autouse=True)
def no_local_settings(monkeypatch, tmp_path):
    # tests see the checked in numbers, not this mac's settings.local.toml
    monkeypatch.setattr(settings, "LOCAL", tmp_path / "settings.local.toml")
