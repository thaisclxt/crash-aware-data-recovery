import random

from dataclasses import dataclass
from typing import List

from ..config import Config
from .Waypoint import Waypoint


@dataclass
class GridEnvironment:
    config: Config
    waypoints: List[Waypoint]
    depot_index: int
    target_waypoints: List[int]

    @classmethod
    def from_config(cls, cfg: Config) -> "GridEnvironment":
        random.seed(cfg.simulation.seed)

        waypoints = cls._build_grid(cfg)
        depot_index = cfg.environment.depot_index
        target_waypoints = cls._select_random_targets(
            total=len(waypoints),
            depot_index=depot_index,
            total_targets=cfg.environment.total_targets,
        )

        env = cls(
            config=cfg,
            waypoints=waypoints,
            depot_index=depot_index,
            target_waypoints=target_waypoints,
        )

        env.assign_random_revenues()
        env.assign_random_risks()

        return env

    @staticmethod
    def _build_grid(cfg: Config) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        width = cfg.grid.width
        height = cfg.grid.height
        spacing = cfg.grid.spacing

        base_revenue = cfg.waypoint.revenue.base
        base_risk = cfg.waypoint.risk.base

        for x in range(width):
            for y in range(height):
                location = (x * spacing, y * spacing)
                waypoints.append(
                    Waypoint(
                        location=location,
                        revenue=base_revenue,
                        risk=base_risk,
                    )
                )

        return waypoints

    @staticmethod
    def _select_random_targets(
        total: int,
        depot_index: int,
        total_targets: int,
    ) -> List[int]:
        candidates = [i for i in range(total) if i != depot_index]
        return random.sample(candidates, total_targets)

    @staticmethod
    def _assign_random_revenues(
        waypoints: List[Waypoint],
        target_indices: List[int],
        min_revenue: float,
        max_revenue: float,
    ) -> None:
        for idx in target_indices:
            waypoints[idx].revenue = random.uniform(min_revenue, max_revenue)

    @staticmethod
    def _assign_random_risks(
        waypoints: List[Waypoint],
        target_indices: List[int],
    ) -> None:
        for idx in target_indices:
            waypoints[idx].risk = random.choice([0.0, 1.0])

    def assign_random_revenues(self) -> None:
        self._assign_random_revenues(
            waypoints=self.waypoints,
            target_indices=self.target_waypoints,
            min_revenue=self.config.waypoint.revenue.min,
            max_revenue=self.config.waypoint.revenue.max,
        )

    def assign_random_risks(self) -> None:
        self._assign_random_risks(
            waypoints=self.waypoints,
            target_indices=self.target_waypoints,
        )
