import random

from typing import Dict, List

from ..config import Config
from ..models.waypoint import Waypoint

from ..utils import print_environment


class GridEnvironment:
    def __init__(self, config: Config) -> None:
        self.config = config

        self.width = config.grid.width
        self.height = config.grid.height
        self.spacing = config.grid.spacing

        self.total_uavs = config.environment.total_uavs
        self.total_targets = config.environment.total_targets
        self.depot_location = config.environment.depot_location

        self.waypoints: List[Waypoint] = self.build_grid()
        self.waypoints_by_id: Dict[int, Waypoint] = {
            wp.w_id: wp for wp in self.waypoints
        }

        self.target_waypoints: List[Waypoint] = self.select_random_targets()

        self.assign_random_revenues()
        self.assign_random_risks()

    def build_grid(self) -> List[Waypoint]:
        waypoints: List[Waypoint] = []

        w_id = 0
        for x in range(self.width):
            for y in range(self.height):
                location = (x * self.spacing, y * self.spacing)

                if location == self.depot_location:
                    continue

                waypoints.append(
                    Waypoint(
                        w_id=w_id,
                        location=location,
                        revenue=self.config.waypoint.revenue.base,
                        risk=self.config.waypoint.risk.base,
                    )
                )
                w_id += 1

        return waypoints

    def get_waypoint(self, w_id: int) -> Waypoint:
        return self.waypoints_by_id[w_id]

    def select_random_targets(self) -> List[Waypoint]:
        return random.sample(self.waypoints, self.total_targets)

    def assign_random_revenues(self) -> None:
        min_revenue = self.config.waypoint.revenue.min
        max_revenue = self.config.waypoint.revenue.max

        for wp in self.target_waypoints:
            wp.revenue = random.uniform(min_revenue, max_revenue)

        # print_environment(self)

    def assign_random_risks(self) -> None:
        for wp in self.target_waypoints:
            wp.risk = random.choice([0, 1])

        # print_environment(self)
