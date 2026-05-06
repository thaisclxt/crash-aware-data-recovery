from typing import List, Optional, Tuple

from ..config import Config
from ..environment.grid_environment import GridEnvironment
from ..models.waypoint import Waypoint
from ..utils import euclidean_distance, calculate_max_cycles


class UAV:
    def __init__(self, uav_id: int, sequence: List[int], config: Config) -> None:
        self.uav_id = uav_id
        self.sequence = sequence
        self.config = config

        # State variables
        self.remaining_flight_time: float = config.uav.max_flight_time
        self.health: float = config.mdp.state.health.default
        self.link_quality: float = config.mdp.state.link_quality.default
        self.collected_revenue: float = config.mdp.state.collected_revenue.default

        self.last_event_time: float = 0.0
        self.accumulated_risk: int = 0
        self.accumulated_revenue: float = 0.0 # currently carried this mission
        self.delivered_revenue: float = 0.0 # lifetime delivered across missions
        self.backed_up_revenue: float = 0.0 # current mission only
        self.total_backed_up_revenue: float = 0.0 # lifetime across missions

        self.active: bool = True
        self.final_status = "active"

        self.completed_cycles: int = 0
        self.max_cycles: int = 1

        self.sequence_index: int = -1
        self.current_waypoint_id: Optional[int] = None

        self.travel_origin: Optional[Tuple[float, float]] = None
        self.travel_destination: Optional[Tuple[float, float]] = None
        self.travel_start_time: Optional[float] = None
        self.travel_end_time: Optional[float] = None

        self.completed_missions: int = 0


    def update_max_cycles(self, env: GridEnvironment) -> None:
        self.max_cycles = calculate_max_cycles(self, env, self.config)


    def current_waypoint(self, env: GridEnvironment) -> Optional[Waypoint]:
        if self.current_waypoint_id is None:
            return None
        return env.get_waypoint(self.current_waypoint_id)


    def start_travel(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        start_time: float,
        end_time: float,
    ) -> None:
        self.travel_origin = origin
        self.travel_destination = destination
        self.travel_start_time = start_time
        self.travel_end_time = end_time


    def finish_travel(self) -> None:
        self.travel_origin = None
        self.travel_destination = None
        self.travel_start_time = None
        self.travel_end_time = None


    def is_traveling(self, now: float) -> bool:
        return (
            self.travel_origin is not None
            and self.travel_destination is not None
            and self.travel_start_time is not None
            and self.travel_end_time is not None
            and self.travel_start_time <= now < self.travel_end_time
        )


    def current_location(
        self,
        env: GridEnvironment,
        now: Optional[float] = None,
    ) -> Tuple[float, float]:
        if now is not None and self.is_traveling(now):
            ox, oy = self.travel_origin
            dx, dy = self.travel_destination

            duration = self.travel_end_time - self.travel_start_time
            if duration <= 0:
                return self.travel_destination

            progress = (now - self.travel_start_time) / duration
            progress = max(0.0, min(1.0, progress))

            x = ox + progress * (dx - ox)
            y = oy + progress * (dy - oy)
            return (x, y)

        wp = self.current_waypoint(env)
        if wp is None:
            return self.config.environment.depot_location
        return wp.location


    def peek_next_waypoint_id(self) -> Optional[int]:
        next_index = self.sequence_index + 1
        if next_index >= len(self.sequence):
            return None
        return self.sequence[next_index]


    def advance_to_next_waypoint(self) -> Optional[int]:
        next_waypoint_id = self.peek_next_waypoint_id()
        if next_waypoint_id is None:
            return None

        self.sequence_index += 1
        self.current_waypoint_id = next_waypoint_id
        return next_waypoint_id


    def has_finished_sequence(self) -> bool:
        return self.peek_next_waypoint_id() is None


    def update_accumulated_risk(self, env: GridEnvironment) -> int:
        wp = self.current_waypoint(env)
        if wp is None:
            return 0

        self.accumulated_risk += wp.risk
        return wp.risk


    def update_accumulated_revenue(self, env: GridEnvironment) -> float:
        wp = self.current_waypoint(env)
        if wp is None:
            return 0.0

        if wp.location == self.config.environment.depot_location:
            return 0.0

        self.accumulated_revenue += wp.revenue
        return wp.revenue


    def update_health(
        self,
        max_flight_time: float,
        max_wp_risk: float,
        alpha: float,
        beta: float,
        good_threshold: float,
        warning_threshold: float,
    ) -> None:
        flight_fraction = (max_flight_time - self.remaining_flight_time) / max_flight_time
        flight_fraction = max(0.0, min(1.0, flight_fraction))

        risk_fraction = 0.0
        if max_wp_risk > 0.0:
            risk_fraction = min(1.0, self.accumulated_risk / max_wp_risk)

        score = alpha * (1.0 - flight_fraction) + beta * (1.0 - risk_fraction)

        if score > good_threshold:
            self.health = 1.0
        elif score > warning_threshold:
            self.health = 0.5
        else:
            self.health = 0.0


    def update_link_quality(
        self,
        env: GridEnvironment,
        all_uavs: List["UAV"],
        communication_range: float,
        now: float,
    ) -> None:
        indicators: List[int] = []

        my_location = self.current_location(env, now)

        for other in all_uavs:
            if other.uav_id == self.uav_id or not other.active:
                continue

            other_location = other.current_location(env, now)
            dist = euclidean_distance(my_location, other_location)
            indicators.append(1 if dist <= communication_range else 0)

        self.link_quality = sum(indicators) / len(indicators) if indicators else 0.0


    def update_collected_revenue(
        self,
        total_targets: int,
        max_wp_revenue: float,
        low_threshold: float,
        medium_threshold: float,
    ) -> None:
        max_possible_revenue = total_targets * max_wp_revenue
        if max_possible_revenue <= 0.0:
            self.collected_revenue = 0.0
            return

        # Count both carried and backed-up revenue for this mission
        mission_revenue = self.accumulated_revenue + self.backed_up_revenue
        revenue_fraction = min(1.0, mission_revenue / max_possible_revenue)

        if revenue_fraction < low_threshold:
            self.collected_revenue = 0.0
        elif revenue_fraction < medium_threshold:
            self.collected_revenue = 0.5
        else:
            self.collected_revenue = 1.0
    
    
    def health_label(self) -> str:
        good_threshold = self.config.mdp.state.health.threshold.good
        warning_threshold = self.config.mdp.state.health.threshold.warning

        if self.health >= good_threshold:
            return "good"
        if self.health >= warning_threshold:
            return "warning"
        return "critical"


    def collected_revenue_label(self) -> str:
        medium_threshold = self.config.mdp.state.collected_revenue.threshold.medium
        low_threshold = self.config.mdp.state.collected_revenue.threshold.low

        if self.collected_revenue >= medium_threshold:
            return "high"
        if self.collected_revenue >= low_threshold:
            return "medium"
        return "low"
