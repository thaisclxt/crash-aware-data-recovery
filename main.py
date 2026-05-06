import random

from pathlib import Path

from src.config import load_configuration
from src.simulator.simulator import Simulator

from src.environment.grid_environment import GridEnvironment
from src.allocation.task_allocator import TaskAllocator
from src.policy.backup_policy import BackupPolicy


def main() -> None:
    config = load_configuration(Path("settings.yaml"))

    random.seed(config.simulation.seed)

    env = GridEnvironment(config=config)

    allocator = TaskAllocator(env=env)
    allocator.initialize_uavs()
    allocator.assign_random_sequences()

    policy = BackupPolicy(config=config)

    sim = Simulator(
        config=config,
        env=env,
        uavs=allocator.uavs,
        policy=policy,
    )

    sim.run()
    sim.save_results("output/results.xlsx")


if __name__ == "__main__":
    main()