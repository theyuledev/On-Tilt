import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.contracts import Intent
from src.inputs import hand_process, hand_view
from src.inputs.hand import HandIntent
from src.inputs.hand_view import JPEG_PERIOD_S, JPEG_WIDTH, draw, frame_jpg, publish
from src.io.ports import Frame
from src.io.settings import load
from src.web.server import make_app

pytestmark = pytest.mark.unit

CFG = {"hand_span": 0.25, "neutral_s": 1.0, "neutral_keep_s": 4.0, "smoothing": 0.0}
LANDMARKS = [SimpleNamespace(x=0.3 + 0.01 * i, y=0.6 - 0.015 * i, z=0.0) for i in range(21)]
INTENT = Intent(0.4, -0.1, True, 1.0)


def test_draw_shrinks_to_a_mirrored_view_with_the_skeleton():
    img = np.zeros((720, 1280, 3), np.uint8)
    out = draw(img, LANDMARKS, (0.5, 0.5), (12.0, -3.0), INTENT)
    assert out.shape == (360, JPEG_WIDTH, 3) and out.any()
    assert not img.any()
    # the hand sits left of centre in the camera image, so right of centre on the page
    skeleton = out[100:, :] > 0
    assert skeleton[:, JPEG_WIDTH // 2 :].sum() > skeleton[:, : JPEG_WIDTH // 2].sum()


def test_draw_without_a_hand_says_so():
    out = draw(np.zeros((480, 640, 3), np.uint8), None, None, None, INTENT)
    assert out.shape == (480, 640, 3) and out.any()


def test_frame_jpg_reads_the_model_result_with_or_without_a_hand():
    img = np.zeros((480, 640, 3), np.uint8)
    for lms in ([], [LANDMARKS]):
        result = SimpleNamespace(hand_landmarks=lms, hand_world_landmarks=lms)
        jpg = frame_jpg(img, result, None, INTENT)
        assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape == img.shape


def test_publish_replaces_the_file_whole(tmp_path):
    path = tmp_path / "hand.jpg"
    publish(b"one", path)
    publish(b"two", path)
    assert path.read_bytes() == b"two" and os.listdir(tmp_path) == ["hand.jpg"]


@pytest.fixture
def served(tmp_path, monkeypatch):
    path = tmp_path / "hand.jpg"
    monkeypatch.setattr(hand_view, "HAND_JPG", path)
    return path, TestClient(make_app(SimpleNamespace(cfg=load())))


def test_hand_jpg_404s_when_missing(served):
    assert served[1].get("/hand.jpg").status_code == 404


def test_hand_jpg_404s_when_the_hand_process_stopped_writing(served):
    path, client = served
    path.write_bytes(b"jpg")
    old = time.time() - load()["game"]["heartbeat_stale_s"] - 1
    os.utime(path, (old, old))
    assert client.get("/hand.jpg").status_code == 404


def test_hand_jpg_serves_a_fresh_file(served):
    path, client = served
    path.write_bytes(b"jpg")
    r = client.get("/hand.jpg")
    assert r.status_code == 200 and r.content == b"jpg"
    assert r.headers["content-type"] == "image/jpeg"


STEP_S = 0.75 * JPEG_PERIOD_S  # the view should land on every other frame


class Frames:
    def __init__(self, n):
        self.left = [Frame(np.full((2, 2), i, np.uint8), i * STEP_S) for i in range(n)]

    @property
    def running(self):
        return bool(self.left)

    def read(self):
        return self.left.pop(0)


def run_hand(view=False):
    events = []
    times = iter(i * STEP_S for i in range(100))
    show = {"show": lambda img, intent: events.append(("show", int(img[0, 0]), intent))}
    hand_process.run(
        Frames(10),
        lambda img, t: (0.5 + t / 10, 0.5),
        HandIntent({**CFG, "neutral_s": 2 * STEP_S}),
        lambda intent, **flags: events.append(("send", intent, flags)),
        clock=lambda: next(times),
        **(show if view else {}),
    )
    return events


def test_the_hand_view_leaves_the_sent_intent_alone_and_runs_at_the_encode_rate():
    events = run_hand(view=True)
    sent = [e for e in events if e[0] == "send"]
    assert sent == run_hand() and any(e[1].valid for e in sent)
    shows = [i for i, e in enumerate(events) if e[0] == "show"]
    assert [events[i][1] for i in shows] == [0, 2, 4, 6, 8]
    # each view comes right after the send for its frame and shows that intent
    assert all(events[i - 1][0] == "send" and events[i - 1][1] == events[i][2] for i in shows)


def test_page_shows_the_hand_cam_above_the_results_and_retries_it():
    client = TestClient(make_app(SimpleNamespace(cfg=load())))
    page = client.get("/").text
    assert all(f'id="{i}"' in page for i in ("handview", "handcam", "handnone", "handtilt"))
    assert page.index('id="side"') < page.index('id="handview"') < page.index('id="results"')
    js = client.get("/static/app.js").text
    assert '"/hand.jpg?"' in js and "handRetry = setTimeout(nextHand, 1000)" in js
    assert '$("handtilt").textContent = `plate x ${p.sent_x_deg' in js


def test_a_broken_view_never_stops_the_intents(capsys):
    sent, tries = [], []

    def broken(img, intent):
        tries.append(intent)
        raise OSError("disk full")

    times = iter(i * STEP_S for i in range(100))
    hand_process.run(
        Frames(10),
        lambda img, t: (0.5, 0.5),
        HandIntent(CFG),
        lambda intent, **flags: sent.append(intent),
        clock=lambda: next(times),
        show=broken,
    )
    assert len(sent) == 10 and len(tries) == 5
    assert capsys.readouterr().out == "hand view off: OSError('disk full')\n"
