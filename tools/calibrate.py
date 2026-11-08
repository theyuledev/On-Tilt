import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from src.io.settings import ROOT, env, load
from src.vision.calib import (
    CALIB_PATH,
    find_corners,
    fit_lens,
    fit_plate,
    is_new_view,
    load_calib,
    plate_size_mm,
    save_calib,
)

VIEWS = ROOT / "data" / "calib" / "views"
USAGE = "usage: python -m tools.calibrate plate|capture|fit|lens-set fx fy cx cy k1 k2 p1 p2 k3"


def say(msg: str) -> None:
    print(msg, flush=True)


def write(calib: dict, out: Path) -> dict:
    calib = {**calib, "measured": date.today().isoformat()}
    save_calib(calib, out)
    return calib


def grab(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read()
    except OSError as e:
        raise SystemExit(f"no frame from {url} ({e}), is the app running?") from e


def decode(jpg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def lens_set(values: list[str], out: Path) -> dict:
    if len(values) != 9:
        raise SystemExit(USAGE)
    fx, fy, cx, cy, *dist = map(float, values)
    lens = {"model": "standard", "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "D": dist}
    lens = write(lens, out)
    say(f"lens -> {out}, run plate next")
    return lens


def capture(vision: dict, next_jpg, views: Path) -> int:
    pattern, want = vision["board_corners"], vision["calib_views"]
    views.mkdir(parents=True, exist_ok=True)
    for old in views.glob("*.jpg"):
        old.unlink()
    kept: list[np.ndarray] = []
    end = time.monotonic() + vision["calib_timeout_s"]
    say(f"move the sheet all over the view, tilt it some too. want {want} views")
    while len(kept) < want and time.monotonic() < end:
        jpg = next_jpg()
        corners = find_corners(decode(jpg), pattern)
        if corners is None or not is_new_view(corners, kept, pattern):
            continue
        kept.append(corners)
        (views / f"{len(kept):02d}.jpg").write_bytes(jpg)
        say(f"view {len(kept)}/{want}")
    say(f"got {len(kept)} views")
    return len(kept)


def fit(vision: dict, views: Path, out: Path) -> dict:
    pattern = vision["board_corners"]
    found, size = [], None
    for path in sorted(views.glob("*.jpg")):
        image = cv2.imread(str(path))
        corners = find_corners(image, pattern)
        if corners is not None:
            found.append(corners)
            size = image.shape[1::-1]
    if not found:
        raise SystemExit(f"no board views in {views}, run capture first")
    fits = fit_lens(found, pattern, vision["board_square_mm"], size)
    for f in fits:
        say(f"{f['model']}: {f['rms_px']:.3f} px rms over {len(found)} views")
    say(f"kept {fits[0]['model']} -> {out}, run plate next")
    return write(fits[0], out)


def plate(vision: dict, next_jpg, out: Path) -> dict:
    if not out.exists():
        raise SystemExit(f"no lens in {out}, run lens-set or capture first")
    pattern = vision["board_corners"]
    end = time.monotonic() + vision["calib_timeout_s"]
    corners = None
    while corners is None:
        if time.monotonic() > end:
            raise SystemExit("sheet not found")
        image = decode(next_jpg())
        corners = find_corners(image, pattern)
    calib = write(fit_plate(load_calib(out), corners, pattern, vision["board_square_mm"]), out)
    say(f"plate: {calib['plate_rms_px']:.3f} px, {calib['plate_rms_mm']:.3f} mm rms -> {out}")
    size = plate_size_mm(image, calib)
    say(f"plate edges {size[0]:.0f} x {size[1]:.0f} mm" if size else "plate edges not found")
    return calib


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    vision = load()["vision"]
    url = f"http://127.0.0.1:{env('PORT', '8000')}/camera.jpg?full=1"
    if step == "plate":
        plate(vision, lambda: grab(url), CALIB_PATH)
    elif step == "lens-set":
        lens_set(sys.argv[2:], CALIB_PATH)
    elif step == "capture":
        capture(vision, lambda: grab(url), VIEWS)
        fit(vision, VIEWS, CALIB_PATH)
    elif step == "fit":
        fit(vision, VIEWS, CALIB_PATH)
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
