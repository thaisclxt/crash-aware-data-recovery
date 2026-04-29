from dataclasses import dataclass, field
from typing import List, Optional

from ..config import Config
from ..utils import euclidean_distance


from .GridEnvironment import GridEnvironment


@dataclass
class UAV:
    uav_id: int

    sequence: List[int] = field(default_factory=list)
    current_index: int = 0
    last_event_time: float = 0.0

    accumulated_risk: float = 0.0
    accumulated_revenue: float = 0.0
    remaining_flight_time: float = 0.0

    health: float = 1.0
    link_quality: float = 1.0
    collected_revenue: float = 0.0

    delivered_revenue: float = 0.0
    backed_up_revenue: float = 0.0

    returning_to_depot: bool = False
    active: bool = True

    def initialize_from_config(self, config: Config) -> None:
        self.remaining_flight_time = config.uav.max_flight_time

    def current_wp_index(self) -> int:
        return self.sequence[self.current_index]

    def next_wp_index(self) -> Optional[int]:
        if self.current_index + 1 < len(self.sequence):
            return self.sequence[self.current_index + 1]
        return None

    def update_accumulated_risk(self, env: GridEnvironment) -> float:
        wp_idx = self.current_wp_index()
        wp = env.waypoints[wp_idx]
        self.accumulated_risk += wp.risk
        return wp.risk

    def update_accumulated_revenue(self, env: GridEnvironment, depot_index: int) -> float:
        wp_idx = self.current_wp_index()
        if wp_idx == depot_index:
            return 0.0

        wp = env.waypoints[wp_idx]
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
    ) -> None:
        indicators: List[int] = []
        my_wp = env.waypoints[self.current_wp_index()]

        for other in all_uavs:
            if other.uav_id == self.uav_id or not other.active:
                continue

            other_wp = env.waypoints[other.current_wp_index()]
            dist = euclidean_distance(my_wp, other_wp)
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

        revenue_fraction = min(1.0, self.accumulated_revenue / max_possible_revenue)

        if revenue_fraction < low_threshold:
            self.collected_revenue = 0.0
        elif revenue_fraction < medium_threshold:
            self.collected_revenue = 0.5
        else:
            self.collected_revenue = 1.0
