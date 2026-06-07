import heapq
import random
import pandas as pd

from typing import Dict, List, Optional

from ..config import Config
from ..environment.grid_environment import GridEnvironment
from ..models.uav import UAV
from ..policy.backup_policy import BackupPolicy
from ..utils import travel_time
from .event import Event


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

        self.event_log: List[Dict] = []
        self.waypoint_log: List[Dict] = []
        self.allocation_log: List[Dict] = []
        self.metrics_log: List[Dict] = []

        self.handlers = {
            "arrival_at_wp": self._handle_arrival_at_wp,
            "arrival_at_depot": self._handle_arrival_at_depot,
            "departure_from_wp": self._handle_departure_from_wp,
            "departure_from_depot": self._handle_departure_from_depot,
            "update_wp_risk": self._handle_update_wp_risk,
            "update_wp_revenue": self._handle_update_wp_revenue,
        }


    def schedule(
        self,
        time: float,
        event_type: str,
        uav_id: Optional[int] = None
    ) -> None:
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
            if self.time < self.config.simulation.time_limit:
                self._log_state(uav=uav, event_type="arrival", wp_id="depot")
                self.schedule(self.time + uav.preparation_time, "departure_from_depot", uav.uav_id)

        self.schedule(self.config.waypoint.risk.update_interval, "update_wp_risk")
        self.schedule(self.config.waypoint.revenue.update_interval, "update_wp_revenue")


    def run(self) -> None:
        self._log_allocation()
        self._log_waypoint_update("initial")
        
        self.schedule_initial_events()

        while self.events and self.time < self.config.simulation.time_limit:
            event = heapq.heappop(self.events)
            self.time = event.time
            self._dispatch(event)

        self._log_uav_metrics()


    def _dispatch(self, event: Event) -> None:
        handler = self.handlers.get(event.event_type)
        if handler is None:
            raise ValueError(f"Unknown event type: {event.event_type}")
        handler(event)


    def _handle_arrival_at_wp(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)

        self._update_uav_time(uav)
        uav.finish_travel()

        crash_draw, crashed = self._check_geometric_crash(uav)
        crash_prob = self.config.uav.base_crash_probability

        if crashed:
            self._crash_uav(uav, crash_draw=crash_draw, crash_prob=crash_prob)
            return

        risk_added = uav.update_accumulated_risk(self.env)
        revenue_added = uav.update_accumulated_revenue(self.env)

        self._log_state(
            uav=uav, 
            event_type="arrival",
            crash_draw=crash_draw,
            crash_prob=crash_prob
        )

        self.schedule(
            self.time + uav.hover_time,
            "departure_from_wp",
            uav.uav_id,
        )


    def _handle_departure_from_wp(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)

        self._update_uav_decision_state(uav)

        action = self.policy.decide_action(uav)
        uav.metrics.record_action(action)

        self._log_state(uav=uav, event_type="departure", action=action)

        if action == "backup":
            self._handle_backup(uav)
        elif action == "continue":
            self._continue_in_tour(uav)
        else:
            raise ValueError(f"Unknown action: {action}")


    def _handle_departure_from_depot(self, event):
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)
        
        # keep this so we can update the link quality that is good by default because in case of only one UAV, it will be wrong
        self._update_uav_decision_state(uav)

        next_wp_id = uav.peek_next_wp_id()

        depot = self.config.environment.depot_location
        next_wp = self.env.get_waypoint(next_wp_id)

        dt = travel_time(depot, next_wp.location, uav.speed)

        self._log_state(uav=uav, event_type="departure", wp_id="depot")

        uav.start_travel(
            origin=depot,
            destination=next_wp.location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        uav.advance_in_tour()
        self.schedule(self.time + dt, "arrival_at_wp", uav.uav_id)


    def _continue_in_tour(self, uav: UAV) -> None:
        next_wp_id = uav.peek_next_wp_id()

        if next_wp_id is None:
            return

        current_location = uav.current_location(self.env, self.time)

        if next_wp_id == "depot":
            depot_location = self.config.environment.depot_location
            dt = travel_time(current_location, depot_location, self.config.uav.speed)

            uav.start_travel(
                origin=current_location,
                destination=depot_location,
                start_time=self.time,
                end_time=self.time + dt,
            )

            uav.advance_in_tour()
            self.schedule(self.time + dt, "arrival_at_depot", uav.uav_id)
            return

        next_wp = self.env.get_waypoint(next_wp_id)
        dt = travel_time(current_location, next_wp.location, self.config.uav.speed)

        uav.start_travel(
            origin=current_location,
            destination=next_wp.location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        uav.advance_in_tour()
        self.schedule(self.time + dt, "arrival_at_wp", uav.uav_id)


    def _handle_backup(self, uav: UAV) -> None:
        backup_amount = uav.accumulated_revenue

        uav.metrics.record_backup(revenue=backup_amount, success=True)

        uav.backed_up_revenue += backup_amount
        uav.total_backed_up_revenue += backup_amount

        # reset accumulated revenue after backup
        uav.accumulated_revenue = 0.0

        self._continue_in_tour(uav)


    def _handle_arrival_at_depot(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)

        self._update_uav_time(uav)
        uav.finish_travel()

        delivered_amount = uav.accumulated_revenue + uav.backed_up_revenue
        uav.delivered_revenue += delivered_amount

        self._log_state(uav=uav, event_type="arrival", wp_id="depot")

        uav.metrics.record_completed_tours()
        uav.metrics.record_delivered_revenue(delivered_amount)

        self._reset_uav_for_new_mission(uav)

        if self.time < self.config.simulation.time_limit:
            self.schedule(self.time + uav.preparation_time, "departure_from_depot", uav.uav_id)


    def _reset_uav_for_new_mission(self, uav: UAV) -> None:
        uav.remaining_flight_time = self.config.uav.max_flight_time
        uav.health = self.config.mdp.state.health.default
        uav.link_quality = self.config.mdp.state.link_quality.default
        uav.collected_revenue = self.config.mdp.state.collected_revenue.default

        uav.last_event_time = self.time
        uav.accumulated_risk = 0
        uav.accumulated_revenue = 0.0
        uav.backed_up_revenue = 0.0
        uav.tour_index = 0

        uav.finish_travel()


    def _handle_update_wp_risk(self, event: Event) -> None:
        # call assign target risks again
        self.env.assign_target_risks()

        # log update type = risk
        self._log_waypoint_update("risk")

        # schedule the next wp update type = risk
        self.schedule(self.time + self.config.waypoint.risk.update_interval, "update_wp_risk")


    def _handle_update_wp_revenue(self, event: Event) -> None:
        # call assign target revenue again
        self.env.assign_target_revenues()

        # log update type = revenue
        self._log_waypoint_update("revenue")

        # schedule the next wp update type = revenue
        self.schedule(self.time + self.config.waypoint.revenue.update_interval, "update_wp_revenue")


    def _update_uav_time(self, uav: UAV) -> None:
        elapsed = self.time - uav.last_event_time
        uav.remaining_flight_time -= elapsed
        uav.last_event_time = self.time


    def _check_geometric_crash(self, uav: UAV) -> tuple[float, bool]:
        crash_draw = random.random()
        crash_prob = self.config.uav.base_crash_probability
        crashed = crash_draw < crash_prob
        return crash_draw, crashed


    def _crash_uav(self, uav: UAV, crash_draw: float, crash_prob: float) -> None:
        lost_amount = uav.accumulated_revenue
        uav.lost_revenue += lost_amount

        uav.metrics.record_crash()
        uav.metrics.record_lost_revenue(lost_amount)
        
        self._log_state(
            uav=uav, 
            event_type="crash",
            action="crash",
            crash_draw=crash_draw,
            crash_prob=crash_prob
        )

        self._replace_crashed_uav(uav)


    def _replace_crashed_uav(self, crashed_uav: UAV) -> None:
        if self.time >= self.config.simulation.time_limit:
            return

        new_uav = UAV(
            uav_id=crashed_uav.uav_id,
            sequence=crashed_uav.sequence,
            config=self.config,
        )

        new_uav.delivered_revenue = crashed_uav.delivered_revenue
        new_uav.total_backed_up_revenue = crashed_uav.total_backed_up_revenue
        new_uav.completed_missions = crashed_uav.completed_missions
        new_uav.lost_revenue = crashed_uav.lost_revenue
        new_uav.update_tour_stats(self.env)

        self.uavs[new_uav.uav_id] = new_uav

        self.schedule(self.time + new_uav.preparation_time, "departure_from_depot", new_uav.uav_id)


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
            total_targets=self.env.total_targets,
            max_wp_revenue=self.config.waypoint.revenue.max,
            low_threshold=self.config.mdp.state.collected_revenue.threshold.low,
            medium_threshold=self.config.mdp.state.collected_revenue.threshold.medium,
        )


    def _log_state(
        self,
        uav: UAV,
        event_type: str,
        wp_id: Optional[str] = None,
        action: Optional[str] = None,
        crash_draw: Optional[float] = None,
        crash_prob: Optional[float] = None,
    ) -> None:
        wp = uav.current_waypoint(self.env)

        self.event_log.append({
            "time": self.time,
            "uav_id": uav.uav_id,
            "event_type": event_type,
            "wp_id": wp_id if wp_id is not None else uav.current_waypoint_id,
            "wp_revenue": wp.revenue if wp else None,
            "wp_risk": wp.risk if wp else None,
            "uav_remaining_flight_time": uav.remaining_flight_time,
            "uav_health": uav.health_label(),
            "uav_link_quality": uav.link_quality,
            "uav_collected_revenue": uav.collected_revenue_label(),
            "uav_accumulated_risk": uav.accumulated_risk,
            "uav_accumulated_revenue": uav.accumulated_revenue,
            "uav_backed_up_revenue": uav.backed_up_revenue,
            "uav_delivered_revenue": uav.delivered_revenue,
            "action": action if action else "",
            "crash_draw": crash_draw if crash_draw is not None else "",
            "crash_prob": crash_prob if crash_prob is not None else "",
        })


    def _log_waypoint_update(self, update_type: str) -> None:
        for wp in self.env.target_waypoints:
            self.waypoint_log.append({
                "time": self.time,
                "update_type": update_type,
                "wp_id": wp.w_id,
                "wp_location": str(wp.location),
                "wp_revenue": wp.revenue,
                "wp_risk": wp.risk,
            })
    

    def _log_allocation(self) -> None:
        for uav in self.uavs.values():
            self.allocation_log.append({
                "uav_id": uav.uav_id,
                "sequence": str(uav.sequence),
                "m_j": uav.m_j,
                "tour": str(uav.tour),
                "estimated_tour_time": uav.estimated_tour_time,
                "max_flight_time": uav.max_flight_time,
            })


    def _log_uav_metrics(self) -> None:
        for uav in self.uavs.values():
            self.metrics_log.append({
                "uav_id": uav.uav_id,
                "completed_tours": uav.metrics.completed_tours,
                "total_crashes": uav.metrics.total_crashes,
                "total_lost_revenue": uav.metrics.total_lost_revenue,
                "total_revenue_backed_up": uav.metrics.total_revenue_backed_up,
                "total_delivered_revenue": uav.metrics.total_delivered_revenue,
                "backup_actions": uav.metrics.backup_actions,
                "continue_actions": uav.metrics.continue_actions,
                "successful_backups": uav.metrics.successful_backups,
                "failed_backups": uav.metrics.failed_backups,
            })


    def save_results(self, output_path: str = "simulation_results.xlsx") -> None:
        df_events = pd.DataFrame(self.event_log)
        df_waypoints = pd.DataFrame(self.waypoint_log)
        df_allocations = pd.DataFrame(self.allocation_log)
        df_metrics = pd.DataFrame(self.metrics_log)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_events.to_excel(writer, sheet_name="UAV Events", index=False)
            df_waypoints.to_excel(writer, sheet_name="Waypoints", index=False)
            df_allocations.to_excel(writer, sheet_name="Allocations", index=False)
            df_metrics.to_excel(writer, sheet_name="UAV Metrics", index=False)
