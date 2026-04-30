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

    print(f"{'Waypoint ID':<15} {'Location':<15} {'Revenue':<15} {'Risk':<5}")
    print("-" * 55)

    for wp in sorted(env.target_waypoints, key=lambda wp: wp.w_id):
        print(f"{wp.w_id:<15} {str(wp.location):<15} {wp.revenue:<15.2f} {wp.risk:<5}")

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
        print(f"{key:<35} {value:>10}")


if __name__ == "__main__":
    main()