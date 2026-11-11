import math

import numpy as np

PALM = (0, 5, 9, 13, 17)


def palm_center(landmarks) -> tuple[float, float]:
    xs = [landmarks[i].x for i in PALM]
    ys = [landmarks[i].y for i in PALM]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def palm_angles(world) -> tuple[float, float] | None:
    """Roll and pitch of the plane through the wrist and four finger bases."""
    points = np.array([[world[i].x, world[i].y, world[i].z] for i in PALM], dtype=float)
    if not np.isfinite(points).all():
        return None
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        return None
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    normal = axes[-1]
    # Both hands share the normal that points toward the camera.
    if normal[2] > 0:
        normal = -normal
    return (
        math.degrees(math.atan2(normal[0], -normal[2])),
        math.degrees(math.atan2(normal[1], -normal[2])),
    )
