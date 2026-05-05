from __future__ import annotations

import math
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config
    from src.environment.grid_environment import GridEnvironment
    from src.models.uav import UAV

Location = Tuple[float, float]

def euclidean_distance(a: Location, b: Location) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def travel_time(a: Location, b: Location, speed: float) -> float:
    return euclidean_distance(a, b) / speed


def calculate_max_cycles(uav: UAV, env: GridEnvironment, config: Config) -> int:
    """
    Calculate maximum number of complete cycles (mj) a UAV can perform
    given its flight time budget.
    """
    depot = config.environment.depot_location
    
    if len(uav.sequence) == 0:
        return 0
    
    # Calculate time for one complete sequence cycle
    cycle_distance = 0.0
    current_loc = depot
    
    for wp_id in uav.sequence:
        wp = env.get_waypoint(wp_id)
        cycle_distance += euclidean_distance(current_loc, wp.location)
        current_loc = wp.location
    
    # Time for one cycle (travel + hover at each waypoint)
    one_cycle_time = (cycle_distance / config.uav.speed) + (len(uav.sequence) * config.uav.hover_time)
    
    # Time to return to depot from last waypoint in sequence
    return_distance = euclidean_distance(current_loc, depot)
    return_time = return_distance / config.uav.speed
    
    # Available time for cycles
    available_time = config.uav.max_flight_time - return_time
    
    if one_cycle_time <= 0:
        return 0
    
    # Calculate max complete cycles
    max_cycles = int(available_time / one_cycle_time)

    if max_cycles < 1:
        # Not enough time for even one cycle
        return 0
    
    return max_cycles