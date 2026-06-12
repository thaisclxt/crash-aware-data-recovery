import random

from typing import Dict, List

from ..config import Config
from ..models.waypoint import Waypoint


class GridEnvironment:
    """
    Represents the grid environment where UAVs operate, including waypoints, target waypoints, and depot location.
    """
    def __init__(self, config: Config) -> None:
        self.config = config

        self.width = config.grid.width
        self.height = config.grid.height
        self.spacing = config.grid.spacing

        self.total_uavs = config.environment.total_uavs
        self.depot_location = config.environment.depot_location

        self.waypoints: List[Waypoint] = self._build_grid()
        self.waypoints_by_id: Dict[int, Waypoint] = {
            wp.w_id: wp for wp in self.waypoints}
        self.waypoints_by_location: Dict[tuple[int, int], Waypoint] = {
            wp.location: wp for wp in self.waypoints
        }

        self.target_waypoints = self._build_targets()
        self.total_targets = len(self.target_waypoints)

        self.assign_target_revenues()
        self.assign_target_risks()


    def _build_grid(self) -> List[Waypoint]:
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
    

    def _build_targets(self) -> List[Waypoint]:
        if self.config.simulation.generate_random_targets:
            requested = self.config.environment.total_targets
            available = len(self.waypoints)

            if requested > available:
                raise ValueError(
                    f"Requested {requested} targets, but only {available} waypoints are available."
                )

            return sorted(random.sample(self.waypoints, requested), key=lambda wp: wp.w_id)
        
        fixed_locations = {
            tuple(location) for location in self.config.environment.fixed_targets
        }

        missing = fixed_locations - set(self.waypoints_by_location)
        if missing:
            raise ValueError(f"Fixed target locations not found in grid: {sorted(missing)}")
        
        return sorted(
            [self.waypoints_by_location[loc] for loc in fixed_locations],
            key=lambda wp: wp.w_id,
        )


    def assign_target_revenues(self) -> None:
        min_revenue = self.config.waypoint.revenue.min
        max_revenue = self.config.waypoint.revenue.max

        for wp in self.target_waypoints:
            wp.revenue = random.uniform(min_revenue, max_revenue)


    def assign_target_risks(self) -> None:
        for wp in self.target_waypoints:
            wp.risk = random.choice([0, 1])


    def get_waypoint(self, w_id: int) -> Waypoint:
        return self.waypoints_by_id[w_id]
