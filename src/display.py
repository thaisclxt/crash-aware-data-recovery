from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.grid_environment import GridEnvironment
    from src.models.uav import UAV

def print_environment(env: GridEnvironment) -> None:
    print(f"\n{'Waypoint ID':<15} {'Location':<15} {'Revenue':<15} {'Risk':<5}")
    print("-" * 55)

    for wp in sorted(env.target_waypoints, key=lambda wp: wp.w_id):
        print(f"{wp.w_id:<15} {str(wp.location):<15} {wp.revenue:<15.2f} {wp.risk:<5}")

    print()


def print_uav_metrics(uavs: List[UAV]) -> None:
    print(f"\n{'UAV ID':<10} {'Active':<10} {'Completed Missions':<20} {'Delivered Revenue':<20} {'Backed Up Revenue':<20} {'Remaining Flight':<18}")
    print("-" * 105)

    for uav in sorted(uavs, key=lambda u: u.uav_id):
        completed_missions = getattr(uav, "completed_missions", 0)
        total_backed_up_revenue = getattr(uav, "total_backed_up_revenue", 0.0)

        print(
            f"{uav.uav_id:<10} "
            f"{str(uav.active):<10} "
            f"{completed_missions:<20} "
            f"{uav.delivered_revenue:<20.2f} "
            f"{total_backed_up_revenue:<20.2f} "
            f"{uav.remaining_flight_time:<18.2f}"
        )
