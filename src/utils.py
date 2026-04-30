import math
from typing import Tuple


Location = Tuple[float, float]

def euclidean_distance(a: Location, b: Location) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def travel_time(a: Location, b: Location, speed: float) -> float:
    return euclidean_distance(a, b) / speed
