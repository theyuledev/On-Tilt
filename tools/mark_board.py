"""Board json for the plate and base markers from one still plus the caliper numbers.

Plate marker centres come from the calipers (outer edges to the plate edges), their in-plane
turn from the still through the homography the 4 centres fix. The board's +x is the servo +x
the flat route uses (plate_yaw_deg). Base markers sit on the printed sheet, spaced as printed.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from src.io.settings import ROOT, load
from src.vision.calib import CALIB_PATH, load_calib, undistort
from src.vision.markers import detector, find
from tools.calibrate import decode, grab

OUT = ROOT / "configs" / "board.local.json"
# a marker's own TL TR BR BL with +y up, top edge toward +y
CORNERS = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], np.float64)
# plate corners going counter-clockwise: BL BR TR TL
RING = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
NAMES = ["BL", "BR", "TR", "TL"]


def square(centre, half: float, deg: float) -> np.ndarray:
    """TL TR BR BL of a marker turned deg counterclockwise about its centre, z = 0."""
    a = np.radians(deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = np.asarray(centre, np.float64) + (CORNERS * half) @ rot.T
    return np.c_[xy, np.zeros(4)]


def _turn(ideal, seen) -> float:
    """Degrees seen is turned from ideal, least squares about their centroids."""
    d = seen - seen.mean(0)
    cross = ideal[:, 0] * d[:, 1] - ideal[:, 1] * d[:, 0]
    return float(np.degrees(np.arctan2(cross.sum(), (ideal * d).sum())))


def ring(rays, yaw_deg: float) -> np.ndarray:
    """Indices of the 4 centres at BL BR TR TL. Of the square's 4 turns, the one whose +x edge
    lies closest to the flat route's +x: it turns the y-up image by +yaw, so +x sits at -yaw."""
    up = np.asarray(rays, np.float64) * [1, -1]
    d = up - up.mean(0)
    ccw = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
    a = np.radians(-yaw_deg)
    want = np.array([np.cos(a), np.sin(a)])

    def along(k):
        bl, br, tr, tl = up[np.roll(ccw, -k)]
        e = br - bl + tr - tl
        return e @ want / np.linalg.norm(e)

    return np.roll(ccw, -max(range(4), key=along))


def fit_plate(corners, lens, plate_ids, insets, size_mm, plate_mm, yaw_deg) -> dict:
    """corners: {id: (4, 2) px}; insets: {id: (outer x edge to plate edge, same for y)} mm."""
    missing = [i for i in plate_ids if i not in corners]
    if missing:
        raise ValueError(f"plate markers not seen: {missing}")
    half = size_mm / 2
    rays = {i: undistort(lens, corners[i]) for i in plate_ids}
    ids = [plate_ids[k] for k in ring([rays[i].mean(0) for i in plate_ids], yaw_deg)]
    centres_mm = RING * (plate_mm / 2 - np.array([insets[i] for i in ids]) - half)
    H = cv2.findHomography(np.array([rays[i].mean(0) for i in ids]), centres_mm)[0]
    plate, turn, seen = {}, {}, {}
    for i, c in zip(ids, centres_mm, strict=True):
        mm = cv2.perspectiveTransform(rays[i].reshape(-1, 1, 2), H).reshape(4, 2)
        turn[i] = _turn(CORNERS * half, mm)
        seen[i] = float(np.linalg.norm(np.roll(mm, -1, 0) - mm, axis=1).mean())
        plate[i] = square(c, half, turn[i])
    px = np.array([corners[i].mean(0) for i in ids])
    sides_px = np.linalg.norm(np.roll(px, -1, 0) - px, axis=1)
    sides_mm = np.linalg.norm(np.roll(centres_mm, -1, 0) - centres_mm, axis=1)
    return {
        "plate": {str(i): p.tolist() for i, p in plate.items()},
        "square_side_px": float(sides_px.mean()),
        "px_per_mm_at_plate": float((sides_px / sides_mm).mean()),
        "plate_corner": {str(i): n for i, n in zip(ids, NAMES, strict=True)},
        "marker_rot_deg": {str(i): t for i, t in turn.items()},
        "marker_seen_mm": {str(i): s for i, s in seen.items()},
    }


def base_board(base_ids, size_mm: float, spacing_mm: float) -> dict:
    left, right = base_ids
    pts = {left: square((-spacing_mm / 2, 0), size_mm / 2, 0)}
    pts[right] = square((spacing_mm / 2, 0), size_mm / 2, 0)
    return {str(i): p.tolist() for i, p in pts.items()}


def sheet_spacing_mm(sheet_gray, det, ids, size_mm) -> float:
    """Centre spacing of the base markers on the printed sheet, scaled by the markers' own
    side in sheet px, so no dpi assumption."""
    found = find(det, sheet_gray)
    missing = [i for i in ids if i not in found]
    if missing:
        raise ValueError(f"sheet markers not seen: {missing}")
    side = np.mean([np.linalg.norm(np.roll(found[i], -1, 0) - found[i], axis=1) for i in ids])
    a, b = (found[i].mean(0) for i in ids)
    return float(np.linalg.norm(a - b) / side * size_mm)


def inset(text: str) -> tuple[int, tuple[float, float]]:
    i, _, xy = text.partition("=")
    x, y = (float(v) for v in xy.split(","))
    return int(i), (x, y)


def still(args) -> np.ndarray:
    if args.url:
        image = decode(grab(args.url))
    elif args.clip:
        cap = cv2.VideoCapture(str(args.clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"no frame {args.frame} in {args.clip}")
    else:
        image = cv2.imread(str(args.still))
        if image is None:
            raise SystemExit(f"cannot read {args.still}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def parse(vision: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="write the marker boards from a still and calipers")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--still", type=Path)
    src.add_argument("--url", help="camera.jpg url of the running app")
    src.add_argument("--clip", type=Path)
    p.add_argument("--frame", type=int, default=0)
    edges = "mm from the marker's outer x and y edges to the nearest plate edges"
    p.add_argument("--inset", type=inset, nargs=4, required=True, metavar="ID=X,Y", help=edges)
    size, yaw = vision.get("marker_size_mm"), vision.get("plate_yaw_deg")
    p.add_argument("--size", type=float, default=size, required=size is None, help="marker mm")
    yaw_help = "plate_yaw_deg of the flat route, picks which way the board's +x points"
    p.add_argument("--yaw-deg", type=float, default=yaw, required=yaw is None, help=yaw_help)
    base = p.add_mutually_exclusive_group(required=True)
    base.add_argument("--sheet", type=Path, help="base marker sheet png, spacing read off it")
    base.add_argument("--base-spacing-mm", type=float)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


def spacing(args, det, base_ids) -> float:
    if args.sheet is None:
        return args.base_spacing_mm
    sheet = cv2.imread(str(args.sheet), cv2.IMREAD_GRAYSCALE)
    if sheet is None:
        raise SystemExit(f"cannot read {args.sheet}")
    return sheet_spacing_mm(sheet, det, base_ids, args.size)


def main() -> None:
    cfg = load()
    vision = cfg["vision"]
    plate_ids, base_ids = vision["plate_marker_ids"], vision["base_marker_ids"]
    args = parse(vision)
    if not CALIB_PATH.exists():
        raise SystemExit(f"no lens in {CALIB_PATH}, run python -m tools.calibrate first")
    det = detector(vision)
    found = find(det, still(args))
    for i, c in sorted(found.items()):
        print(f"id {i}: " + " ".join(f"({x:.1f},{y:.1f})" for x, y in c))
    plate_mm, gap = cfg["rig"]["plate_inner_mm"], spacing(args, det, base_ids)
    try:
        out = fit_plate(
            found, load_calib(), plate_ids, dict(args.inset), args.size, plate_mm, args.yaw_deg
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"square side {out['square_side_px']:.1f} px, {out['px_per_mm_at_plate']:.3f} px/mm")
    for i in plate_ids:
        k = str(i)
        rot, seen = out["marker_rot_deg"][k], out["marker_seen_mm"][k]
        print(f"id {i}: {out['plate_corner'][k]}, turned {rot:+.2f} deg, seen as {seen:.1f} mm")
    print(f"base spacing {gap:.2f} mm")
    out |= {"base": base_board(base_ids, args.size, gap), "marker_size_mm": args.size}
    out["measured"] = date.today().isoformat()
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
