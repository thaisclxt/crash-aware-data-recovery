from pathlib import Path

from src.config import load_configuration
from src.simulator import Simulator

from src.models.GridEnvironment import GridEnvironment
from src.models.TaskAllocator import TaskAllocator
from src.models.BackupPolicy import BackupPolicy



def main() -> None:
    cfg = load_configuration(Path("settings.yaml"))

    env = GridEnvironment.from_config(cfg)

    allocator = TaskAllocator.from_environment(env)
    allocator.assign_random_sequences()

    for uav in allocator.uavs:
        print(f"UAV {uav.uav_id} sequence: {uav.sequence}")

    policy = BackupPolicy(config=cfg)

    sim = Simulator(
        config=cfg,
        env=env,
        uavs=allocator.uavs,
        policy=policy,
    )
    sim.run()


if __name__ == "__main__":
    main()