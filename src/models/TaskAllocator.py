import random

from dataclasses import dataclass, field
from typing import List

from .GridEnvironment import GridEnvironment
from .UAV import UAV


@dataclass
class TaskAllocator:
    env: GridEnvironment
    uavs: List[UAV] = field(default_factory=list)

    @classmethod
    def from_environment(cls, env: GridEnvironment) -> "TaskAllocator":
        allocator = cls(env=env)
        allocator.initialize_uavs()
        return allocator

    def initialize_uavs(self) -> None:
        self.uavs.clear()

        total_uavs = self.env.config.environment.total_uavs
        depot_index = self.env.depot_index

        for uid in range(1, total_uavs + 1):
            uav = UAV(
                uav_id=uid,
                sequence=[depot_index],
            )
            uav.initialize_from_config(self.env.config)
            self.uavs.append(uav)

    def assign_random_sequences(self) -> None:
        if not self.uavs:
            raise ValueError("UAVs must be initialized before assigning sequences.")

        targets = list(self.env.target_waypoints)
        random.shuffle(targets)

        for i, wp_idx in enumerate(targets):
            self.uavs[i % len(self.uavs)].sequence.append(wp_idx)
