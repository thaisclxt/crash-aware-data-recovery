import heapq

from typing import Dict, List, Optional

from ..config import Config
from ..environment.grid_environment import GridEnvironment
from ..models.uav import UAV
from ..policy.backup_policy import BackupPolicy
from ..utils import travel_time, calculate_max_cycles, print_environment
from .event import Event
from .metrics import Metrics


class Simulator:
    def __init__(
        self,
        config: Config,
        env: GridEnvironment,
        uavs: List[UAV],
        policy: BackupPolicy,
    ) -> None:
        self.config = config
        self.env = env
        self.uavs: Dict[int, UAV] = {uav.uav_id: uav for uav in uavs}
        self.policy = policy

        self.time: float = 0.0
        self.event_counter: int = 0
        self.events: List[Event] = []
        self.metrics = Metrics()

        self.handlers = {
            "uav_arrival": self._handle_uav_arrival,
            "uav_departure": self._handle_uav_departure,
            "update_wp_risk": self._handle_update_wp_risk,
            "update_wp_revenue": self._handle_update_wp_revenue,
            "depot_arrival": self._handle_depot_arrival,
        }

    def schedule(self, time: float, event_type: str, uav_id: Optional[int] = None) -> None:
        self.event_counter += 1
        event = Event(
            time=time,
            priority=self.event_counter,
            event_type=event_type,
            uav_id=uav_id,
        )
        heapq.heappush(self.events, event)

    def get_uav(self, uav_id: int) -> UAV:
        return self.uavs[uav_id]

    def schedule_initial_events(self) -> None:
        for uav in self.uavs.values():
            self._schedule_initial_arrival(uav)

        self.schedule(self.config.waypoint.risk.update_interval, "update_wp_risk")
        self.schedule(self.config.waypoint.revenue.update_interval, "update_wp_revenue")

    def run(self) -> None:
        # Calculate max_cycles for each UAV
        for uav in self.uavs.values():
            uav.max_cycles = calculate_max_cycles(uav, self.env, self.config)
            self._log(f"UAV {uav.uav_id} planned for {uav.max_cycles} cycles")

        self.schedule_initial_events()

        while self.events and self.time < self.config.simulation.time_limit:
            event = heapq.heappop(self.events)
            self.time = event.time
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        handler = self.handlers.get(event.event_type)
        if handler is None:
            raise ValueError(f"Unknown event type: {event.event_type}")

        handler(event)

    def _schedule_initial_arrival(self, uav: UAV) -> None:
        first_wp_id = uav.peek_next_waypoint_id()
        if first_wp_id is None:
            uav.active = False
            return

        depot_location = self.config.environment.depot_location
        first_wp = self.env.get_waypoint(first_wp_id)

        dt = travel_time(depot_location, first_wp.location, self.config.uav.speed)

        uav.start_travel(
            origin=depot_location,
            destination=first_wp.location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        uav.advance_to_next_waypoint()
        self.schedule(self.time + dt, "uav_arrival", uav.uav_id)

    def _handle_uav_arrival(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)
        if not uav.active:
            return

        self._log(f"UAV {uav.uav_id} arrived at WP {uav.current_waypoint_id}")

        if not self._advance_uav_time(uav):
            return
        
        uav.finish_travel()

        risk_added = uav.update_accumulated_risk(self.env)
        revenue_added = uav.update_accumulated_revenue(self.env)

        self.metrics.record_risk_accumulated(risk_added)
        self.metrics.record_revenue_collected(revenue_added)

        self.schedule(
            self.time + self.config.uav.hover_time,
            "uav_departure",
            uav.uav_id,
        )

    def _handle_uav_departure(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)
        if not uav.active:
            return

        self._log(f"UAV {uav.uav_id} departed from WP {uav.current_waypoint_id}")

        if not self._advance_uav_time(uav):
            return

        self._update_uav_decision_state(uav)

        if hasattr(uav, "print_states"):
            uav.print_states()

        action = self.policy.decide_action(uav)
        self.metrics.record_action(action)

        if action == "backup":
            self._handle_backup(uav)
            return
        elif action == "continue":
            self._continue_to_next_waypoint(uav)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _handle_update_wp_risk(self, event: Event) -> None:
        self.env.assign_random_risks()

        print_environment(self.env)

        self.schedule(
            self.time + self.config.waypoint.risk.update_interval,
            "update_wp_risk",
        )

    def _handle_update_wp_revenue(self, event: Event) -> None:
        self.env.assign_random_revenues()

        print_environment(self.env)

        self.schedule(
            self.time + self.config.waypoint.revenue.update_interval,
            "update_wp_revenue",
        )

    def _advance_uav_time(self, uav: UAV) -> bool:
        elapsed = self.time - uav.last_event_time
        uav.remaining_flight_time -= elapsed

        if uav.remaining_flight_time <= 0:
            uav.active = False
            self.metrics.record_crash()
            self._log(f"UAV {uav.uav_id} crashed")
            return False

        uav.last_event_time = self.time
        return True

    def _update_uav_decision_state(self, uav: UAV) -> None:
        uav.update_health(
            max_flight_time=self.config.uav.max_flight_time,
            max_wp_risk=self.config.waypoint.risk.max,
            alpha=self.config.mdp.state.health.alpha,
            beta=self.config.mdp.state.health.beta,
            good_threshold=self.config.mdp.state.health.threshold.good,
            warning_threshold=self.config.mdp.state.health.threshold.warning,
        )

        uav.update_link_quality(
            env=self.env,
            all_uavs=list(self.uavs.values()),
            communication_range=self.config.uav.communication_range,
            now=self.time,
        )

        uav.update_collected_revenue(
            total_targets=self.config.environment.total_targets,
            max_wp_revenue=self.config.waypoint.revenue.max,
            low_threshold=self.config.mdp.state.collected_revenue.threshold.low,
            medium_threshold=self.config.mdp.state.collected_revenue.threshold.medium,
        )

    def _continue_to_next_waypoint(self, uav: UAV) -> None:
        next_wp_id = uav.peek_next_waypoint_id()

        if next_wp_id is None:
            uav.completed_cycles += 1

            self._log(
                f"UAV {uav.uav_id} completed cycle "
                f"{uav.completed_cycles}/{uav.max_cycles}"
            )

            if uav.completed_cycles >= uav.max_cycles:
                self._log(
                    f"UAV {uav.uav_id} completed all {uav.max_cycles} cycles, "
                    f"returning to depot"
                )
                self._return_to_depot(uav)
                return

            uav.sequence_index = -1
            uav.current_waypoint_id = None
            next_wp_id = uav.peek_next_waypoint_id()

            if next_wp_id is None:
                self._log(f"UAV {uav.uav_id} has empty sequence, returning to depot")
                self._return_to_depot(uav)
                return

            self._log(
                f"UAV {uav.uav_id} starting cycle "
                f"{uav.completed_cycles + 1}/{uav.max_cycles}"
            )

        current_location = uav.current_location(self.env, self.time)
        next_wp = self.env.get_waypoint(next_wp_id)
        dt = travel_time(current_location, next_wp.location, self.config.uav.speed)

        uav.start_travel(
            origin=current_location,
            destination=next_wp.location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        uav.advance_to_next_waypoint()
        self.schedule(self.time + dt, "uav_arrival", uav.uav_id)

    def _handle_backup(self, uav: UAV) -> None:
        self._log(f"UAV {uav.uav_id} selected action: backup")

        self.metrics.record_backup(
            revenue=uav.accumulated_revenue,
            success=True,
        )

        uav.backed_up_revenue += uav.accumulated_revenue
        uav.accumulated_revenue = 0.0

        self._continue_to_next_waypoint(uav)

    def _return_to_depot(self, uav: UAV) -> None:
        current_location = uav.current_location(self.env, self.time)
        depot_location = self.config.environment.depot_location

        dt = travel_time(current_location, depot_location, self.config.uav.speed)

        uav.start_travel(
            origin=current_location,
            destination=depot_location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        uav.returning_to_depot = True
        self.schedule(self.time + dt, "depot_arrival", uav.uav_id)
    

    def _handle_depot_arrival(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)
        if not uav.active:
            return

        if not self._advance_uav_time(uav):
            return

        uav.finish_travel()

        delivered_amount = uav.accumulated_revenue + uav.backed_up_revenue
        uav.delivered_revenue += delivered_amount

        self._log(
            f"UAV {uav.uav_id} arrived at depot and delivered "
            f"{delivered_amount:.2f}"
        )

        uav.accumulated_revenue = 0.0
        uav.backed_up_revenue = 0.0
        uav.accumulated_risk = 0.0
        uav.returning_to_depot = False
        uav.active = False

        self.metrics.record_completed_mission()

    def _log(self, message: str) -> None:
        print(f"[{self.time:.1f}] {message}")
