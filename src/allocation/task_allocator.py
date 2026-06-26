import random

from typing import List

from ..environment.grid_environment import GridEnvironment
from ..models.uav import UAV


class TaskAllocator:
    """
    A simple task allocator that assigns target waypoints to UAVs randomly.
    """
    def __init__(self, env: GridEnvironment, total_uavs: int) -> None:
        self.env = env
        self.total_uavs = total_uavs
        self.uavs = self._build_uavs()

        self._assign_tasks()


    def _build_uavs(self) -> List[UAV]:
        return [
            UAV(
                uav_id=uav_id,
                sequence=[],
                config=self.env.config,
            )
            for uav_id in range(self.total_uavs)
        ]


    def _assign_tasks(self) -> None:
        """
        Randomly assign target waypoints to UAVs and compute tour stats.
        """
        target_ids = [wp.w_id for wp in self.env.target_waypoints]
        random.shuffle(target_ids)

        for index, target_id in enumerate(target_ids):
            uav = self.uavs[index % len(self.uavs)]
            uav.sequence.append(target_id)

        for uav in self.uavs:
            uav.prepare_mission(self.env)
