from dataclasses import dataclass
from typing import Tuple


@dataclass
class Waypoint:
    w_id: int
    location: Tuple[float, float]
    revenue: float
    risk: float

    @property
    def x(self) -> float:
        return self.location[0]

    @property
    def y(self) -> float:
        return self.location[1]
