import random

from typing import List

from ..environment.grid_environment import GridEnvironment
from ..models.uav import UAV


class TaskAllocator:
    def __init__(self, env: GridEnvironment) -> None:
        self.env = env
        self.uavs: List[UAV] = []


    def initialize_uavs(self) -> List[UAV]:
        self.uavs = [
            UAV(
                uav_id=uav_id,
                sequence=[],
                config=self.env.config,
            )
            for uav_id in range(self.env.total_uavs)
        ]
        return self.uavs


    def assign_random_sequences(self) -> None:
        if not self.uavs:
            raise ValueError("UAVs must be initialized before assigning sequences.")

        unassigned_targets = [wp.w_id for wp in self.env.target_waypoints]

        while unassigned_targets:
            for uav in self.uavs:
                if not unassigned_targets:
                    break

                target_id = random.choice(unassigned_targets)
                uav.sequence.append(target_id)
                unassigned_targets.remove(target_id)

    def get_uav_sequences(self) -> List[str]:
        return [
            f"UAV {uav.uav_id} sequence: {uav.sequence}"
            for uav in self.uavs
        ]
