import heapq

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import Config
from .models.BackupPolicy import BackupPolicy
from .models.GridEnvironment import GridEnvironment
from .models.UAV import UAV
from .utils import euclidean_distance, travel_time


@dataclass
class Simulator:
    config: Config
    env: GridEnvironment
    uavs: List[UAV]
    policy: BackupPolicy

    time: float = 0.0
    events: List[Tuple[float, int, str, Optional[int]]] = field(default_factory=list)
    event_counter: int = 0
    metrics: Dict[str, float] = field(default_factory=lambda: {
        "backup_actions": 0,
        "return_actions": 0,
        "continue_actions": 0,
        "crashes": 0,
        "depot_deliveries": 0,
        "successful_backups": 0,
        "revenue_delivered_to_depot": 0.0,
        "revenue_backed_up": 0.0,
    })

    def schedule(self, time: float, event_type: str, uav_id: Optional[int] = None) -> None:
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_type, uav_id))

    def get_uav(self, uav_id: int) -> UAV:
        return next(u for u in self.uavs if u.uav_id == uav_id)

    def schedule_initial_events(self) -> None:
        for uav in self.uavs:
            self.schedule(0.0, "uav_arrival", uav.uav_id)
        self.schedule(0.0, "update_wp_risk")
        self.schedule(0.0, "update_wp_revenue")

    def attempt_backup(self, source: UAV) -> bool:
        source_wp = self.env.waypoints[source.current_wp_index()]
        candidates: List[Tuple[float, UAV]] = []

        for other in self.uavs:
            if other.uav_id == source.uav_id or not other.active:
                continue

            other_wp = self.env.waypoints[other.current_wp_index()]
            dist = euclidean_distance(source_wp, other_wp)
            if dist <= self.config.uav.communication_range:
                candidates.append((other.remaining_flight_time, other))

        if not candidates:
            return False

        _, receiver = max(candidates, key=lambda x: x[0])
        amount = source.accumulated_revenue
        receiver.backed_up_revenue += amount
        self.metrics["successful_backups"] += 1
        self.metrics["revenue_backed_up"] += amount

        print(
            f"[{self.time:.1f}] UAV {source.uav_id} backed up "
            f"{amount:.1f} revenue to UAV {receiver.uav_id}"
        )
        return True

    def replan_to_depot(self, uav: UAV) -> None:
        current_idx = uav.current_wp_index()
        if current_idx == self.env.depot_index:
            return

        uav.sequence = [current_idx, self.env.depot_index]
        uav.current_index = 0
        uav.returning_to_depot = True

    def deliver_at_depot(self, uav: UAV) -> None:
        delivered = uav.accumulated_revenue + uav.backed_up_revenue
        if delivered > 0:
            uav.delivered_revenue += delivered
            self.metrics["revenue_delivered_to_depot"] += delivered
            self.metrics["depot_deliveries"] += 1

            print(f"[{self.time:.1f}] UAV {uav.uav_id} delivered {delivered:.1f} revenue at depot")

        uav.accumulated_revenue = 0.0
        uav.backed_up_revenue = 0.0
        uav.accumulated_risk = 0.0
        uav.remaining_flight_time = self.config.uav.max_flight_time
        uav.health = 1.0
        uav.collected_revenue = 0.0
        uav.returning_to_depot = False
        uav.active = False

    def run(self) -> None:
        self.schedule_initial_events()

        sim_time_limit = self.config.simulation.time
        total_targets = self.config.environment.total_targets
        max_wp_revenue = self.config.waypoint.revenue.max
        wp_risk_update_dt = self.config.waypoint.risk.update_interval
        wp_revenue_update_dt = self.config.waypoint.revenue.update_interval

        uav_hover_time = self.config.uav.hover_time
        comm_range = self.config.uav.communication_range

        health_cfg = self.config.mdp.state.health
        collected_rev_cfg = self.config.mdp.state.collected_revenue
        risk_max = self.config.waypoint.risk.max

        while self.events and self.time < sim_time_limit:
            self.time, _, event_type, uav_id = heapq.heappop(self.events)

            if event_type == "uav_arrival" and uav_id is not None:
                uav = self.get_uav(uav_id)
                if not uav.active:
                    continue

                wp_idx = uav.current_wp_index()
                print(f"[{self.time:.1f}] UAV {uav.uav_id} arrived at WP {wp_idx}")

                elapsed = self.time - uav.last_event_time
                uav.remaining_flight_time -= elapsed
                if uav.remaining_flight_time <= 0:
                    self.metrics["crashes"] += 1
                    uav.active = False
                    print(f"[{self.time:.1f}] UAV {uav.uav_id} crashed")
                    continue

                uav.last_event_time = self.time

                if wp_idx == self.env.depot_index and uav.returning_to_depot:
                    self.deliver_at_depot(uav)
                    continue

                uav.update_accumulated_risk(self.env)
                uav.update_accumulated_revenue(self.env, self.env.depot_index)

                self.schedule(self.time + uav_hover_time, "uav_departure", uav.uav_id)

            elif event_type == "uav_departure" and uav_id is not None:
                uav = self.get_uav(uav_id)
                if not uav.active:
                    continue

                wp_idx = uav.current_wp_index()
                print(f"[{self.time:.1f}] UAV {uav.uav_id} departed from WP {wp_idx}")

                elapsed = self.time - uav.last_event_time
                uav.remaining_flight_time -= elapsed
                if uav.remaining_flight_time <= 0:
                    self.metrics["crashes"] += 1
                    uav.active = False
                    print(f"[{self.time:.1f}] UAV {uav.uav_id} crashed")
                    continue

                uav.update_health(
                    max_flight_time=self.config.uav.max_flight_time,
                    max_wp_risk=risk_max,
                    alpha=health_cfg.alpha,
                    beta=health_cfg.beta,
                    good_threshold=health_cfg.threshold_good,
                    warning_threshold=health_cfg.threshold_warning,
                )
                uav.update_link_quality(
                    env=self.env,
                    all_uavs=self.uavs,
                    communication_range=comm_range,
                )
                uav.update_collected_revenue(
                    total_targets=total_targets,
                    max_wp_revenue=max_wp_revenue,
                    low_threshold=collected_rev_cfg.threshold_low,
                    medium_threshold=collected_rev_cfg.threshold_medium,
                )

                print(
                    f"[{self.time:.1f}] UAV {uav.uav_id} state -> "
                    f"health: {uav.health}, "
                    f"link: {uav.link_quality:.2f}, "
                    f"revenue_state: {uav.collected_revenue}"
                )

                action = self.policy.decide_action(uav)

                if action == "return_to_depot":
                    self.metrics["return_actions"] += 1
                    self.replan_to_depot(uav)
                    print(f"[{self.time:.1f}] UAV {uav.uav_id} action: RETURN_TO_DEPOT")
                elif action == "backup":
                    self.metrics["backup_actions"] += 1
                    ok = self.attempt_backup(uav)
                    if ok:
                        print(f"[{self.time:.1f}] UAV {uav.uav_id} action: BACKUP")
                    else:
                        print(
                            f"[{self.time:.1f}] UAV {uav.uav_id} action: "
                            "BACKUP requested but no neighbor available"
                        )
                else:
                    self.metrics["continue_actions"] += 1
                    print(f"[{self.time:.1f}] UAV {uav.uav_id} action: CONTINUE")

                next_idx = uav.next_wp_index()
                if next_idx is None:
                    print(f"[{self.time:.1f}] UAV {uav.uav_id} finished sequence")
                    uav.active = False
                    continue

                curr_wp = self.env.waypoints[uav.current_wp_index()]
                next_wp = self.env.waypoints[next_idx]
                dt = travel_time(curr_wp, next_wp, self.config.uav.speed)

                uav.current_index += 1
                uav.last_event_time = self.time
                self.schedule(self.time + dt, "uav_arrival", uav.uav_id)

            elif event_type == "update_wp_risk":
                self.env.assign_random_risks()
                self.schedule(self.time + wp_risk_update_dt, "update_wp_risk")

            elif event_type == "update_wp_revenue":
                self.env.assign_random_revenues()
                self.schedule(self.time + wp_revenue_update_dt, "update_wp_revenue")

        self.print_summary()

    def print_summary(self) -> None:
        print("\n===== Simulation Summary =====")
        print(f"Continue actions: {int(self.metrics['continue_actions'])}")
        print(f"Backup actions: {int(self.metrics['backup_actions'])}")
        print(f"Return-to-depot actions: {int(self.metrics['return_actions'])}")
        print(f"Successful backups: {int(self.metrics['successful_backups'])}")
        print(f"Depot deliveries: {int(self.metrics['depot_deliveries'])}")
        print(f"Crashes: {int(self.metrics['crashes'])}")
        print(f"Revenue backed up: {self.metrics['revenue_backed_up']:.1f}")
        print(f"Revenue delivered to depot: {self.metrics['revenue_delivered_to_depot']:.1f}")
