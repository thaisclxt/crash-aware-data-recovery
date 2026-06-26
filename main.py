import random

from pathlib import Path

from src.config import load_configuration
from src.simulator.simulator import Simulator

from src.environment.grid_environment import GridEnvironment
from src.allocation.task_allocator import TaskAllocator
from src.policy.backup_policy import BackupPolicy


def main() -> None:
    # All configuration parameters are loaded from the settings.yaml file
    config = load_configuration(Path("settings.yaml"))

    # Set the random seed for reproducibility
    random.seed(config.simulation.seed)

    # Initialize the environment, task allocator, and policy
    env = GridEnvironment(config=config)
    
    count=0
    # Run the simulation for different cb and number of uavs values
    for cb in config.uav.base_crash_probability:
        for total_uavs in config.environment.total_uavs:

            # Create a simple task allocator that randomly assigns target waypoints to UAVs
            allocator = TaskAllocator(env=env, total_uavs=total_uavs)

            # Create the backup policy object
            policy = BackupPolicy(config=config)

            # Run the simulation for an specified number of time and save the results to an Excel file
            sim = Simulator(
                config=config,
                env=env,
                uavs=allocator.uavs,
                policy=policy,
                cb=cb,
            )

            sim.run()

            count+=1
            sim.save_results(f"results/{cb}crash{total_uavs}uavs.xlsx")


if __name__ == "__main__":
    main()
