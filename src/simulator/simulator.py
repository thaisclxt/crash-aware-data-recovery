import heapq
import random
import pandas as pd

from typing import Dict, List, Optional

from ..config import Config
from ..environment.grid_environment import GridEnvironment
from ..models.uav import UAV
from ..policy.backup_policy import BackupPolicy
from ..utils import travel_time
from ..allocation.tour_planner import is_depot
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

        self.event_log: List[Dict] = []
        self.waypoint_log: List[Dict] = []
        self.allocation_log: List[Dict] = []
        self.assignment_log: List[Dict] = []

        self._initialize_uavs()

        self.handlers = {
            "uav_arrival": self._handle_uav_arrival,
            "uav_departure": self._handle_uav_departure,
            "update_wp_risk": self._handle_update_wp_risk,
            "update_wp_revenue": self._handle_update_wp_revenue,
            "depot_arrival": self._handle_depot_arrival,
        }


    def _initialize_uavs(self) -> None:
        for uav in self.uavs.values():
            uav.update_tour_stats(self.env)


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
            if uav.active and self.time < self.config.simulation.time_limit:
                self._schedule_initial_departure(uav)

        self.schedule(self.config.waypoint.risk.update_interval, "update_wp_risk")
        self.schedule(self.config.waypoint.revenue.update_interval, "update_wp_revenue")


    def run(self) -> None:
        self._log_allocation()
        self._log_uav_metrics()
        self._log_waypoint_update("initial")
        self.schedule_initial_events()

        while self.events and self.time < self.config.simulation.time_limit:
            event = heapq.heappop(self.events)
            self.time = event.time
            self._dispatch(event)

        for uav in self.uavs.values():
            if uav.active:
                uav.final_status = "operational"

        self._finalize_assignment_log()


    def _dispatch(self, event: Event) -> None:
        handler = self.handlers.get(event.event_type)
        if handler is None:
            raise ValueError(f"Unknown event type: {event.event_type}")
        handler(event)


    def _schedule_initial_departure(self, uav: UAV) -> None:
        next_node_id = uav.peek_next_node_id()
        if next_node_id is None:
            uav.active = False
            uav.final_status = "empty_tour"
            return

        if is_depot(next_node_id):
            uav.active = False
            uav.final_status = "degenerate_tour"
            return

        depot_location = self.config.environment.depot_location
        next_wp = self.env.get_waypoint(next_node_id)
        dt = travel_time(depot_location, next_wp.location, self.config.uav.speed)

        uav.start_travel(
            origin=depot_location,
            destination=next_wp.location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        self._log_state(uav, "depot_departure", wp_id="depot")

        uav.advance_in_tour()
        self.schedule(self.time + dt, "uav_arrival", uav.uav_id)


    def _handle_uav_arrival(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)
        if not uav.active:
            return

        self._update_uav_time(uav)
        uav.finish_travel()

        crash_draw, crashed = self._check_geometric_crash(uav)
        crash_prob = self.config.uav.base_crash_probability

        if crashed:
            self._crash_uav(uav, crash_draw=crash_draw, crash_prob=crash_prob)
            return

        risk_added = uav.update_accumulated_risk(self.env)
        revenue_added = uav.update_accumulated_revenue(self.env)

        self.metrics.record_risk_accumulated(risk_added)
        self.metrics.record_revenue_collected(revenue_added)

        self._log_state(uav, "arrival", crash_draw=crash_draw, crash_prob=crash_prob)

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


        self._update_uav_decision_state(uav)

        action = self.policy.decide_action(uav)
        self.metrics.record_action(action)

        self._log_state(uav, "departure", action)

        if action == "backup":
            self._handle_backup(uav)
        elif action == "continue":
            self._continue_in_tour(uav)
        else:
            raise ValueError(f"Unknown action: {action}")


    def _continue_in_tour(self, uav: UAV) -> None:
        next_node_id = uav.peek_next_node_id()

        if next_node_id is None:
            uav.active = False
            uav.final_status = "tour_finished"
            return

        current_location = uav.current_location(self.env, self.time)

        if is_depot(next_node_id):
            depot_location = self.config.environment.depot_location
            dt = travel_time(current_location, depot_location, self.config.uav.speed)

            uav.start_travel(
                origin=current_location,
                destination=depot_location,
                start_time=self.time,
                end_time=self.time + dt,
            )

            uav.advance_in_tour()
            self.schedule(self.time + dt, "depot_arrival", uav.uav_id)
            return

        next_wp = self.env.get_waypoint(next_node_id)
        dt = travel_time(current_location, next_wp.location, self.config.uav.speed)

        uav.start_travel(
            origin=current_location,
            destination=next_wp.location,
            start_time=self.time,
            end_time=self.time + dt,
        )

        uav.advance_in_tour()
        self.schedule(self.time + dt, "uav_arrival", uav.uav_id)


    def _handle_backup(self, uav: UAV) -> None:
        backup_amount = uav.accumulated_revenue

        assignment_by_id = {row["uav_id"]: row for row in self.assignment_log}
        assignment_by_id[uav.uav_id]["backup_count"] += 1

        self.metrics.record_backup(revenue=backup_amount, success=True)

        uav.backed_up_revenue += backup_amount
        uav.total_backed_up_revenue += backup_amount
        uav.accumulated_revenue = 0.0

        self._continue_in_tour(uav)


    def _handle_depot_arrival(self, event: Event) -> None:
        if event.uav_id is None:
            return

        uav = self.get_uav(event.uav_id)
        if not uav.active:
            return

        self._update_uav_time(uav)
        uav.finish_travel()

        delivered_amount = uav.accumulated_revenue + uav.backed_up_revenue
        uav.delivered_revenue += delivered_amount

        self._log_state(uav, "depot_arrival", wp_id="depot")

        uav.completed_missions += 1
        self.metrics.record_completed_mission()

        self._reset_uav_for_new_mission(uav)

        if self.time < self.config.simulation.time_limit and uav.active:
            self._schedule_initial_departure(uav)


    def _reset_uav_for_new_mission(self, uav: UAV) -> None:
        uav.remaining_flight_time = self.config.uav.max_flight_time
        uav.health = self.config.mdp.state.health.default
        uav.link_quality = self.config.mdp.state.link_quality.default
        uav.collected_revenue = self.config.mdp.state.collected_revenue.default

        uav.last_event_time = self.time
        uav.accumulated_risk = 0
        uav.accumulated_revenue = 0.0
        uav.backed_up_revenue = 0.0

        uav.active = True
        uav.final_status = "operational"

        uav.update_tour_stats(self.env)
        uav.finish_travel()


    def _handle_update_wp_risk(self, event: Event) -> None:
        self.env.assign_random_risks()
        self._log_waypoint_update("risk")
        self.schedule(self.time + self.config.waypoint.risk.update_interval, "update_wp_risk")


    def _handle_update_wp_revenue(self, event: Event) -> None:
        self.env.assign_random_revenues()
        self._log_waypoint_update("revenue")
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
        uav.active = False
        uav.final_status = "crashed"

        lost_amount = uav.accumulated_revenue
        uav.lost_revenue += lost_amount

        assignment_by_id = {row["uav_id"]: row for row in self.assignment_log}
        assignment_by_id[uav.uav_id]["crash_count"] += 1
        assignment_by_id[uav.uav_id]["lost_revenue"] += lost_amount

        self.metrics.record_crash()
        self.metrics.record_lost_revenue(lost_amount)
        
        self._log_state(uav, "crash", crash_draw=crash_draw, crash_prob=crash_prob)

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

        if new_uav.active:
            self._schedule_initial_departure(new_uav)

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
        action: Optional[str] = None,
        wp_id: Optional[str] = None,
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
            "health": uav.health_label(),
            "link_quality": uav.link_quality,
            "collected_revenue": uav.collected_revenue_label(),
            "revenue_fraction": uav.revenue_fraction,
            "accumulated_risk": uav.accumulated_risk,
            "accumulated_revenue": uav.accumulated_revenue,
            "backed_up_revenue": uav.backed_up_revenue,
            "remaining_flight_time": uav.remaining_flight_time,
            "action": action if action else "",
            "crash_draw": crash_draw if crash_draw is not None else "",
            "crash_prob": crash_prob if crash_prob is not None else "",
        })

    def _log_waypoint_update(self, update_type: str) -> None:
        for wp in self.env.target_waypoints:
            self.waypoint_log.append({
                "time": self.time,
                "update_type": update_type,
                "waypoint_id": wp.w_id,
                "location": str(wp.location),
                "revenue": wp.revenue,
                "risk": wp.risk,
            })

    def _log_allocation(self) -> None:
        for uav in self.uavs.values():
            uav.update_tour_stats(self.env)

            self.allocation_log.append({
                "uav_id": uav.uav_id,
                "sequence": str(uav.sequence),
                "m_j": uav.m_j,
                "tour": str(uav.tour),
            })

    def _log_uav_metrics(self) -> None:
        for uav in self.uavs.values():
            uav.update_tour_stats(self.env)

            self.assignment_log.append({
                "uav_id": uav.uav_id,
                "crash_count": 0,
                "backup_count": 0,
                "lost_revenue": 0.0,
            })

    def _finalize_assignment_log(self) -> None:
        assignment_by_id = {row["uav_id"]: row for row in self.assignment_log}

        for uav in self.uavs.values():
            row = assignment_by_id[uav.uav_id]

            if row["crash_count"] > 0 and uav.active:
                final_status = "crashed_and_replaced"
            elif row["crash_count"] > 0 and not uav.active:
                final_status = "crashed"
            elif uav.active:
                final_status = "operational"
            else:
                final_status = uav.final_status

            row.update({
                "completed_missions": uav.completed_missions,
                "delivered_revenue": uav.delivered_revenue,
                "lost_revenue": uav.lost_revenue,
                "avg_revenue_per_mission": uav.delivered_revenue / max(1, uav.completed_missions),
                "total_backed_up_revenue": uav.total_backed_up_revenue,
                "final_status": final_status,
            })


    def _build_summary_metrics(self) -> Dict:
        total_delivered_revenue = sum(uav.delivered_revenue for uav in self.uavs.values())
        total_lost_revenue = sum(getattr(uav, "lost_revenue", 0.0) for uav in self.uavs.values())
        total_backups = sum(row["backup_count"] for row in self.assignment_log)
        total_completed_missions = sum(uav.completed_missions for uav in self.uavs.values())
        total_crashes = sum(row["crash_count"] for row in self.assignment_log)

        backup_rate = (
            total_backups / total_completed_missions
            if total_completed_missions > 0 else 0.0
        )

        revenue_preservation_percent = (
            total_delivered_revenue / (total_delivered_revenue + total_lost_revenue) * 100
            if (total_delivered_revenue + total_lost_revenue) > 0 else 0.0
        )

        return {
            "total_delivered_revenue": total_delivered_revenue,
            "total_lost_revenue": total_lost_revenue,
            "total_backups": total_backups,
            "total_completed_missions": total_completed_missions,
            "total_crashes": total_crashes,
            "backup_rate": backup_rate,
            "revenue_preservation_percent": revenue_preservation_percent,
        }


    def save_results(self, output_path: str = "simulation_results.xlsx") -> None:
        df_events = pd.DataFrame(self.event_log)
        df_waypoints = pd.DataFrame(self.waypoint_log)
        df_allocations = pd.DataFrame(self.allocation_log)
        df_assignments = pd.DataFrame(self.assignment_log)
        df_summary = pd.DataFrame([self._build_summary_metrics()])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_events.to_excel(writer, sheet_name="UAV Events", index=False)
            df_waypoints.to_excel(writer, sheet_name="Waypoints", index=False)
            df_allocations.to_excel(writer, sheet_name="Allocations", index=False)
            df_assignments.to_excel(writer, sheet_name="UAV Metrics", index=False)
            df_summary.to_excel(writer, sheet_name="Summary Metrics", index=False)
