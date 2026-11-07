from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.io.settings import load
from src.web.server import make_app

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    return TestClient(make_app(SimpleNamespace(cfg=load())))


def test_page_has_ball_heartbeat_and_board_lost(client):
    page = client.get("/").text
    assert 'id="ball"' in page
    assert 'id="beat"' in page
    assert 'id="board"' in page


def test_config_gives_plate_ball_and_stale_limit_from_settings(client):
    cfg = load()
    assert client.get("/config").json() == {
        "plate_inner_mm": cfg["rig"]["plate_inner_mm"],
        "ball_d_mm": cfg["rig"]["ball_d_mm"],
        "heartbeat_stale_s": cfg["game"]["heartbeat_stale_s"],
    }
