import random

from pathlib import Path

from src.config import load_configuration
from src.utils import print_environment
from src.simulator.simulator import Simulator

from src.environment.grid_environment import GridEnvironment
from src.allocation.task_allocator import TaskAllocator
from src.policy.backup_policy import BackupPolicy


def main() -> None:
    config = load_configuration(Path("settings.yaml"))

    random.seed(config.simulation.seed)

    env = GridEnvironment(config=config)

    print_environment(env)

    allocator = TaskAllocator(env=env)
    allocator.initialize_uavs()
    allocator.assign_random_sequences()

    print()
    for sequences in allocator.get_uav_sequences():
        print(sequences)

    policy = BackupPolicy(config=config)

    print(f"\n===== Starting simulation with {config.environment.total_targets} targets and {config.environment.total_uavs} UAVs =====\n")

    sim = Simulator(
        config=config,
        env=env,
        uavs=allocator.uavs,
        policy=policy,
    )
    sim.run()

    print(f"\n{'Metrics':<35} {'Value':>10}")
    print("-" * 55)

    for key, value in vars(sim.metrics).items():
        print(f"{key:<35} {value:>10.2f}")


if __name__ == "__main__":
    main()