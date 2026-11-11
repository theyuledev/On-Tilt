import math


class Rest:
    """The dead zone round a balance target: inside it the plate stops pulling and only brakes,
    so friction can park the ball. Every correction there only sets it rocking again."""

    def __init__(self) -> None:
        self.on = False
        self.still_s = 0.0

    def zone_mm(self, goal: tuple[float, float], c: dict) -> float:
        """Radius of the zone, wider the further the target sits from the middle. 0 = no zone."""
        zone = c["balance_rest_mm"]
        return zone + c["balance_rest_grow"] * math.hypot(*goal) if zone else 0.0

    def step(self, far: float, raw_mm_s: float, zone: float, c: dict, edge_mm: float = 1e9) -> bool:
        """Takes the ball's distance from the target. True on the tick it comes into the zone.
        edge_mm is how far the ball is from the edge of the board: closer than
        balance_rest_edge_mm there is no zone, so it may wobble but never off the board.
        Slow is judged on the unsmoothed speed: the smoothed one lags a ball rolling through."""
        was = self.on
        if edge_mm < c["balance_rest_edge_mm"]:
            self.on = False
        elif self.on:
            self.on = far < zone + c["balance_rest_out_mm"] - c["balance_rest_mm"]
        else:
            self.on = far < zone and raw_mm_s < c["balance_rest_mm_s"]
        if not self.on:
            self.still_s = 0.0
        return self.on and not was

    def brake(self, speed: float, dt: float, c: dict) -> bool:
        """False once the ball has lain still long enough: braking a still ball only jitters it.
        The brake comes back when it moves faster than balance_rest_move_mm_s."""
        if speed < c["balance_rest_still_mm_s"]:
            self.still_s += dt
        elif speed > c["balance_rest_move_mm_s"]:
            self.still_s = 0.0
        return not (c["balance_rest_still_mm_s"] and self.still_s >= c["balance_rest_still_s"])


def pull_share(far: float, zone: float, c: dict) -> float:
    """Share of the position error the pull acts on outside the zone: all of it, or with
    balance_rest_soft only the part past the zone's edge, so waking is a nudge and not a yank."""
    if not (c["balance_rest_soft"] and zone and far):
        return 1.0
    return max(far - zone, 0.0) / far
