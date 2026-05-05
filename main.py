import random

from pathlib import Path

from src.config import load_configuration
from src.utils import calculate_max_cycles
from src.display import print_environment, print_uav_metrics
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

    policy = BackupPolicy(config=config)

    print(f"===== Starting simulation with {config.environment.total_targets} targets and {config.environment.total_uavs} UAVs =====\n")

    print(f"{'UAV ID':<10} {'Active':<10} {'Sequence':<20} {'Targets':<15} {'m_j':<12} {'Remaining Flight':<18}")
    print("-" * 90)

    for uav in allocator.uavs:
        uav.max_cycles = calculate_max_cycles(uav, env, config)
        print(
            f"{uav.uav_id:<10} "
            f"{str(uav.active):<10} "
            f"{str(uav.sequence):<20} "
            f"{len(uav.sequence):<15} "
            f"{uav.max_cycles:<12} "
            f"{uav.remaining_flight_time:<18.2f} "
        )

    print()

    sim = Simulator(
        config=config,
        env=env,
        uavs=allocator.uavs,
        policy=policy,
    )
    sim.run()

    sim.save_results("output/results.xlsx")

    print_uav_metrics(sim.uavs.values())

    print(f"\n{'Metrics':<35} {'Value':>10}")
    print("-" * 55)

    for key, value in vars(sim.metrics).items():
        if isinstance(value, float):
            print(f"{key:<35} {value:>10.2f}")
        else:
            print(f"{key:<35} {value:>10}")


if __name__ == "__main__":
    main()