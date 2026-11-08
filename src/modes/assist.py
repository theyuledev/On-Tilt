"""Where the palm points, scaled to a spot the ball can sit at without touching the rim."""

from src.contracts import Intent


def clamp(v: float) -> float:
    return min(max(v, -1.0), 1.0)


def goal_mm(intent: Intent, rig: dict, margin_mm: float) -> tuple[float, float]:
    if not intent.valid:
        return (0.0, 0.0)
    reach = rig["plate_inner_mm"] / 2 - rig["ball_d_mm"] / 2 - margin_mm
    return (reach * clamp(intent.x), reach * clamp(intent.y))
