import math

from .models.Waypoint import Waypoint


def euclidean_distance(wp_a: Waypoint, wp_b: Waypoint) -> float:
    return math.hypot(wp_b.x - wp_a.x, wp_b.y - wp_a.y)


def travel_time(wp_a: Waypoint, wp_b: Waypoint, speed: float) -> float:
    return euclidean_distance(wp_a, wp_b) / speed
