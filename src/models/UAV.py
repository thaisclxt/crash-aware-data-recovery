from typing import List, Optional, Tuple

from ..config import Config
from ..environment.grid_environment import GridEnvironment
from ..models.waypoint import Waypoint
from ..utils import euclidean_distance, travel_time
from ..simulator.metrics import Metrics


class UAV:
    def __init__(self, uav_id: int, sequence: List[int], config: Config) -> None:
        self.uav_id = uav_id
        self.sequence = sequence
        self.config = config

        self.speed = config.uav.speed
        self.max_flight_time = config.uav.max_flight_time
        self.hover_time = config.uav.hover_time
        self.preparation_time = config.uav.preparation_time
        
        self.m_j: int = 0
        self.tour: List[int | str] = []
        self.estimated_tour_time: float = 0.0
        self.remaining_flight_time: float = self.max_flight_time

        self.health: float = config.mdp.state.health.default
        self.link_quality: float = config.mdp.state.link_quality.default
        self.collected_revenue: float = config.mdp.state.collected_revenue.default
        self.backed_up_revenue: float = config.mdp.state.backed_up_revenue.default

        self.last_event_time: float = 0.0

        self.accumulated_risk: int = 0
        self.accumulated_revenue: float = 0.0
        self.delivered_revenue: float = 0.0
        
        self.total_backed_up_revenue: float = 0.0
        self.lost_revenue = 0.0

        self.tour_index: int = 0 # where we are in the tour
        self.current_waypoint_id: Optional[int] = None

        self.travel_origin: Optional[Tuple[float, float]] = None
        self.travel_destination: Optional[Tuple[float, float]] = None
        self.travel_start_time: Optional[float] = None
        self.travel_end_time: Optional[float] = None

        self.metrics = Metrics()


    def prepare_mission(self, env: GridEnvironment) -> None:
        """Compute m_j, estimated_tour_time and set tour indexes"""
        if not self.sequence:
            return
        
        m_j = 0

        depot = self.config.environment.depot_location
        first_target = env.get_waypoint(self.sequence[0]).location
        last_target = env.get_waypoint(self.sequence[-1]).location
        
        # depot -> first target
        x = travel_time(
            origin=depot,
            destination=first_target,
            speed=self.speed
        ) + self.preparation_time

        # last target -> depot
        y = travel_time(
            origin=last_target,
            destination=depot,
            speed=self.speed
        )

        # jump from the end of the sequence back to its beginning
        # in case len(sequence) == 1, k = 0 since there's no travel between targets
        k = travel_time(
            origin=last_target,
            destination=first_target,
            speed=self.speed
        )

        z = self.hover_time
        for wp_id, nxt_wp_id in zip(self.sequence, self.sequence[1:]):
            z += travel_time(
                origin=env.get_waypoint(wp_id).location,
                destination=env.get_waypoint(nxt_wp_id).location,
                speed=self.speed
            ) + self.hover_time

        while True:
            if (x + y + ((m_j+1) * z) + (m_j * k)) > self.max_flight_time:
                break

            time = x + y + ((m_j+1) * z) + (m_j * k)
            m_j += 1

        self.m_j = m_j
        self.estimated_tour_time = time

        self._set_tour()

    
    def _set_tour(self) -> None:
        tour = ["depot"]
        for _ in range(self.m_j):
            for wp_id in self.sequence:
                tour.append(wp_id)
        tour.append("depot")

        self.tour = tour
    

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


    def peek_next_wp_id(self) -> Optional[int | str]:
        next_index = self.tour_index + 1
        if next_index >= len(self.tour):
            return None
        return self.tour[next_index]


    def advance_in_tour(self) -> None:
        next_wp_id = self.peek_next_wp_id()
        if next_wp_id is None or next_wp_id == "depot":
            return None

        self.tour_index += 1
        self.current_waypoint_id = next_wp_id


    def has_finished_tour(self) -> bool:
        return self.tour_index >= len(self.tour) - 1


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

        # alpha and beta are weights that determine the relative importance of flight time and risk in calculating health score
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
            if other.uav_id == self.uav_id:
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

        mission_revenue = self.accumulated_revenue + self.backed_up_revenue
        revenue_fraction = min(1.0, mission_revenue / max_possible_revenue)

        self.revenue_fraction = revenue_fraction  # Store for potential use

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

        if self.collected_revenue > medium_threshold:
            return "high"
        if self.collected_revenue > low_threshold:
            return "medium"
        return "low"
