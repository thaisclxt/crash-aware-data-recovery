from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..config import Config
from ..environment.grid_environment import GridEnvironment
from ..utils import travel_time


Location = Tuple[float, float]
NodeId = int


@dataclass(frozen=True)
class TourStats:
    sequence: List[NodeId]       # S_j
    m_j: int                     # number of periodic visits
    tour: List[NodeId]           # T_j = [depot, S_j repeated m_j times, depot]
    travel_leg_times: List[float]
    hover_times: List[float]     # one per visited node in `tour`
    leg_times: List[float]       # total per step: travel + hover-at-destination
    hover_time_total: float
    tour_time: float

    @property
    def is_empty(self) -> bool:
        return len(self.sequence) == 0 or self.m_j == 0

    @property
    def num_legs(self) -> int:
        return len(self.leg_times)


def _depot_token() -> NodeId:
    return -1


def is_depot(node_id: NodeId) -> bool:
    return node_id == _depot_token()


def resolve_location(
    node_id: NodeId,
    env: GridEnvironment,
    config: Config,
) -> Location:
    if is_depot(node_id):
        return config.environment.depot_location
    return env.get_waypoint(node_id).location


def get_hover_time_for_node(
    node_id: NodeId,
    env: GridEnvironment,
    config: Config,
) -> float:
    """
    Hover time applied when the UAV arrives at `node_id`.

    Convention:
    - Depot has zero hover/service time.
    - Every waypoint visit consumes hover/service time.
    - If your waypoint object has its own service/hover time, use it.
    - Otherwise fall back to a global config value.
    """
    if is_depot(node_id):
        return 0.0

    waypoint = env.get_waypoint(node_id)

    if hasattr(waypoint, "hover_time"):
        return float(waypoint.hover_time)

    if hasattr(waypoint, "service_time"):
        return float(waypoint.service_time)

    if hasattr(config.uav, "hover_time"):
        return float(config.uav.hover_time)

    if hasattr(config.uav, "service_time"):
        return float(config.uav.service_time)

    return 0.0


def build_tour(sequence: List[NodeId], m_j: int) -> List[NodeId]:
    """
    Build the closed tour:
        T_j = [depot, S_j repeated m_j times, depot]
    """
    depot = _depot_token()

    if not sequence or m_j <= 0:
        return [depot, depot]

    return [depot, *(sequence * m_j), depot]


def compute_travel_leg_times(
    tour: List[NodeId],
    env: GridEnvironment,
    config: Config,
) -> List[float]:
    """
    Travel-only time for each leg of the tour.
    Example:
        [d, 1, 2, 3, d] ->
        [time(d,1), time(1,2), time(2,3), time(3,d)]
    """
    if len(tour) < 2:
        return []

    speed = config.uav.speed
    travel_leg_times: List[float] = []

    for i in range(len(tour) - 1):
        origin = resolve_location(tour[i], env, config)
        destination = resolve_location(tour[i + 1], env, config)
        travel_leg_times.append(travel_time(origin, destination, speed))

    return travel_leg_times


def compute_hover_times(
    tour: List[NodeId],
    env: GridEnvironment,
    config: Config,
) -> List[float]:
    """
    Hover/service time associated with arriving at each destination node
    of every leg in the tour.

    Example:
        tour = [d, 1, 2, 3, d]
        destinations are [1, 2, 3, d]
        hover_times = [hover(1), hover(2), hover(3), hover(d)=0]
    """
    if len(tour) < 2:
        return []

    hover_times: List[float] = []

    for i in range(len(tour) - 1):
        destination_node = tour[i + 1]
        hover_times.append(get_hover_time_for_node(destination_node, env, config))

    return hover_times


def compute_leg_times(
    tour: List[NodeId],
    env: GridEnvironment,
    config: Config,
) -> List[float]:
    """
    Total time per leg:
        leg_time = travel_time(origin, destination) + hover_time(destination)
    """
    travel_leg_times = compute_travel_leg_times(tour, env, config)
    hover_times = compute_hover_times(tour, env, config)
    return [travel + hover for travel, hover in zip(travel_leg_times, hover_times)]


def compute_tour_time(
    sequence: List[NodeId],
    m_j: int,
    env: GridEnvironment,
    config: Config,
) -> float:
    """
    Total duration of the full closed tour, including:
    - travel time on every leg
    - hover/service time at every visited waypoint
    """
    tour = build_tour(sequence, m_j)
    leg_times = compute_leg_times(tour, env, config)
    return sum(leg_times)


def compute_m_j(
    sequence: List[NodeId],
    env: GridEnvironment,
    config: Config,
) -> int:
    """
    Find the largest admissible m_j such that the full closed tour
    fits within the UAV maximum flight time.
    """
    if not sequence:
        return 0

    max_flight_time = config.uav.max_flight_time
    m_j = 0

    while True:
        candidate_m_j = m_j + 1
        candidate_tour_time = compute_tour_time(sequence, candidate_m_j, env, config)

        if candidate_tour_time <= max_flight_time:
            m_j = candidate_m_j
        else:
            break

    return m_j


def can_depart_for_full_tour(
    sequence: List[NodeId],
    env: GridEnvironment,
    config: Config,
    remaining_flight_time: float,
) -> bool:
    """
    Explicit departure check:
    the UAV can leave the depot only if it can complete the full closed
    tour and return to the depot with the currently available time.
    """
    if remaining_flight_time <= 0.0:
        return False

    stats = build_tour_stats(sequence, env, config)

    if stats.m_j == 0:
        return False

    return stats.tour_time <= remaining_flight_time


def build_tour_stats(
    sequence: List[NodeId],
    env: GridEnvironment,
    config: Config,
) -> TourStats:
    """
    Build the full closed-tour description for one UAV sequence,
    including travel and hover/service times.
    """
    normalized_sequence = list(sequence)
    m_j = compute_m_j(normalized_sequence, env, config)
    tour = build_tour(normalized_sequence, m_j)

    travel_leg_times = compute_travel_leg_times(tour, env, config)
    hover_times = compute_hover_times(tour, env, config)
    leg_times = [travel + hover for travel, hover in zip(travel_leg_times, hover_times)]

    hover_time_total = sum(hover_times)
    tour_time = sum(leg_times)

    return TourStats(
        sequence=normalized_sequence,
        m_j=m_j,
        tour=tour,
        travel_leg_times=travel_leg_times,
        hover_times=hover_times,
        leg_times=leg_times,
        hover_time_total=hover_time_total,
        tour_time=tour_time,
    )
