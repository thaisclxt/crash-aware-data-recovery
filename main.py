import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ============================================================
# Configuration
# ============================================================

GRID_WIDTH = 13
GRID_HEIGHT = 13
GRID_SPACING = 5.0
DEPOT_INDEX = 0

BASE_WP_REVENUE = 30.0
BASE_WP_RISK = 0.0
MIN_WP_REVENUE = 60.0
MAX_WP_REVENUE = 600.0
MAX_WP_RISK = 6.0
WP_REVENUE_UPDATE_INTERVAL = 20.0
WP_RISK_UPDATE_INTERVAL = 30.0

TOTAL_TARGETS = 21
TOTAL_UAVS = 6

UAV_SPEED = 10.0
UAV_HOVER_TIME = 5.0
MAX_FLIGHT_TIME = 1800.0
COMMUNICATION_RANGE = 60.0

SIM_TIME = 500.0

HEALTH_ALPHA = 0.6
HEALTH_BETA = 0.4
HEALTH_GOOD_THRESHOLD = 0.7
HEALTH_WARNING_THRESHOLD = 0.4

REVENUE_LOW_THRESHOLD = 0.4
REVENUE_MEDIUM_THRESHOLD = 0.7

POLICY_WEIGHT_HEALTH = 0.5
POLICY_WEIGHT_LINK_QUALITY = 0.3
POLICY_WEIGHT_COLLECTED_REVENUE = 0.2

BACKUP_SCORE_THRESHOLD = 0.5
RETURN_LINK_QUALITY_THRESHOLD = 0.15
RETURN_REVENUE_THRESHOLD = 0.5

# ============================================================
# Data models
# ============================================================

@dataclass
class Waypoint:
    x: float
    y: float
    risk: float = BASE_WP_RISK
    revenue: float = BASE_WP_REVENUE


@dataclass
class UAV:
    uav_id: int
    sequence: List[int] = field(default_factory=list)
    current_index: int = 0
    last_event_time: float = 0.0

    accumulated_risk: float = 0.0
    accumulated_revenue: float = 0.0
    remaining_flight_time: float = MAX_FLIGHT_TIME

    health: float = 1.0
    link_quality: float = 1.0
    collected_revenue: float = 0.0

    delivered_revenue: float = 0.0
    backed_up_revenue: float = 0.0

    returning_to_depot: bool = False
    active: bool = True

    def current_wp_index(self) -> int:
        return self.sequence[self.current_index]

    def next_wp_index(self) -> Optional[int]:
        if self.current_index + 1 < len(self.sequence):
            return self.sequence[self.current_index + 1]
        return None

    def update_accumulated_risk(self, env: "GridEnvironment", sim_time: float) -> None:
        wp_idx = self.current_wp_index()
        wp = env.waypoints[wp_idx]
        self.accumulated_risk += wp.risk
        if wp.risk == 1:
            print(f"[{sim_time:.1f}] UAV {self.uav_id} encountered risk at WP {wp_idx}")

    def update_accumulated_revenue(self, env: "GridEnvironment", sim_time: float) -> None:
        wp_idx = self.current_wp_index()
        if wp_idx == DEPOT_INDEX:
            return
        wp = env.waypoints[wp_idx]
        self.accumulated_revenue += wp.revenue
        if wp.revenue > 0:
            print(f"[{sim_time:.1f}] UAV {self.uav_id} collected revenue {wp.revenue:.1f} at WP {wp_idx}")

    def update_health(self) -> None:
        flight_fraction = (MAX_FLIGHT_TIME - self.remaining_flight_time) / MAX_FLIGHT_TIME
        flight_fraction = max(0.0, min(1.0, flight_fraction))
        risk_fraction = min(1.0, self.accumulated_risk / MAX_WP_RISK)
        score = HEALTH_ALPHA * (1.0 - flight_fraction) + HEALTH_BETA * (1.0 - risk_fraction)

        if score > HEALTH_GOOD_THRESHOLD:
            self.health = 1.0
        elif score > HEALTH_WARNING_THRESHOLD:
            self.health = 0.5
        else:
            self.health = 0.0

    def update_link_quality(self, env: "GridEnvironment", all_uavs: List["UAV"]) -> None:
        indicators: List[int] = []
        my_wp = env.waypoints[self.current_wp_index()]

        for other in all_uavs:
            if other.uav_id == self.uav_id or not other.active:
                continue
            other_wp = env.waypoints[other.current_wp_index()]
            dist = euclidean_distance(my_wp, other_wp)
            indicators.append(1 if dist <= COMMUNICATION_RANGE else 0)

        self.link_quality = sum(indicators) / len(indicators) if indicators else 0.0

    def update_collected_revenue(self) -> None:
        max_possible_revenue = TOTAL_TARGETS * MAX_WP_REVENUE
        revenue_fraction = min(1.0, self.accumulated_revenue / max_possible_revenue)

        if revenue_fraction < REVENUE_LOW_THRESHOLD:
            self.collected_revenue = 0.0
        elif revenue_fraction < REVENUE_MEDIUM_THRESHOLD:
            self.collected_revenue = 0.5
        else:
            self.collected_revenue = 1.0


# ============================================================
# Environment
# ============================================================

class GridEnvironment:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.waypoints = self._build_grid()
        self.depot_index = DEPOT_INDEX
        self.target_waypoints = self._select_random_targets()

    def _build_grid(self) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                waypoints.append(Waypoint(x * GRID_SPACING, y * GRID_SPACING))
        return waypoints

    def _select_random_targets(self) -> List[int]:
        candidates = [i for i in range(len(self.waypoints)) if i != self.depot_index]
        return random.sample(candidates, TOTAL_TARGETS)

    def assign_random_revenues(self) -> None:
        for idx in self.target_waypoints:
            self.waypoints[idx].revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)

    def assign_random_risks(self) -> None:
        for idx in self.target_waypoints:
            self.waypoints[idx].risk = random.choice([0, 1])


# ============================================================
# Utilities
# ============================================================

def euclidean_distance(wp_a: Waypoint, wp_b: Waypoint) -> float:
    return math.hypot(wp_b.x - wp_a.x, wp_b.y - wp_a.y)


def travel_time(wp_a: Waypoint, wp_b: Waypoint) -> float:
    return euclidean_distance(wp_a, wp_b) / UAV_SPEED


# ============================================================
# Task allocation
# ============================================================

class TaskAllocator:
    def __init__(self, env: GridEnvironment):
        self.env = env
        self.uavs: List[UAV] = []

    def initialize_uavs(self) -> None:
        for uid in range(1, TOTAL_UAVS + 1):
            self.uavs.append(UAV(uav_id=uid, sequence=[DEPOT_INDEX]))

    def assign_random_sequences(self) -> None:
        targets = list(self.env.target_waypoints)
        random.shuffle(targets)
        for i, wp_idx in enumerate(targets):
            self.uavs[i % len(self.uavs)].sequence.append(wp_idx)


# ============================================================
# Policy
# ============================================================

class BackupPolicy:
    @staticmethod
    def score(uav: UAV) -> float:
        return (
            POLICY_WEIGHT_HEALTH * uav.health
            + POLICY_WEIGHT_LINK_QUALITY * uav.link_quality
            + POLICY_WEIGHT_COLLECTED_REVENUE * uav.collected_revenue
        )

    @staticmethod
    def decide_action(uav: UAV) -> str:
        score = BackupPolicy.score(uav)

        if uav.health == 0.0:
            return "return_to_depot"

        if uav.collected_revenue >= RETURN_REVENUE_THRESHOLD and uav.link_quality <= RETURN_LINK_QUALITY_THRESHOLD:
            return "return_to_depot"

        if score < BACKUP_SCORE_THRESHOLD:
            return "backup"

        return "continue"


# ============================================================
# Simulation engine
# ============================================================

class Simulator:
    def __init__(self, env: GridEnvironment, uavs: List[UAV]):
        self.env = env
        self.uavs = uavs
        self.time = 0.0
        self.events: List[Tuple[float, int, str, Optional[int]]] = []
        self.event_counter = 0

        self.metrics: Dict[str, float] = {
            "backup_actions": 0,
            "return_actions": 0,
            "continue_actions": 0,
            "crashes": 0,
            "depot_deliveries": 0,
            "successful_backups": 0,
            "revenue_delivered_to_depot": 0.0,
            "revenue_backed_up": 0.0,
        }

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
            if dist <= COMMUNICATION_RANGE:
                candidates.append((other.remaining_flight_time, other))

        if not candidates:
            return False

        _, receiver = max(candidates, key=lambda x: x[0])
        amount = source.accumulated_revenue
        receiver.backed_up_revenue += amount
        self.metrics["successful_backups"] += 1
        self.metrics["revenue_backed_up"] += amount
        print(f"[{self.time:.1f}] UAV {source.uav_id} backed up {amount:.1f} revenue to UAV {receiver.uav_id}")
        return True

    def replan_to_depot(self, uav: UAV) -> None:
        current_idx = uav.current_wp_index()
        if current_idx == DEPOT_INDEX:
            return
        uav.sequence = [current_idx, DEPOT_INDEX]
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
        uav.remaining_flight_time = MAX_FLIGHT_TIME
        uav.health = 1.0
        uav.collected_revenue = 0.0
        uav.returning_to_depot = False
        uav.active = False

    def run(self) -> None:
        self.schedule_initial_events()

        while self.events and self.time < SIM_TIME:
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

                if wp_idx == DEPOT_INDEX and uav.returning_to_depot:
                    self.deliver_at_depot(uav)
                    continue

                uav.update_accumulated_risk(self.env, self.time)
                uav.update_accumulated_revenue(self.env, self.time)
                self.schedule(self.time + UAV_HOVER_TIME, "uav_departure", uav.uav_id)

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

                uav.update_health()
                uav.update_link_quality(self.env, self.uavs)
                uav.update_collected_revenue()

                print(
                    f"[{self.time:.1f}] UAV {uav.uav_id} state -> health: {uav.health}, "
                    f"link: {uav.link_quality:.2f}, revenue_state: {uav.collected_revenue}"
                )

                action = BackupPolicy.decide_action(uav)

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
                        print(f"[{self.time:.1f}] UAV {uav.uav_id} action: BACKUP requested but no neighbor available")
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
                dt = travel_time(curr_wp, next_wp)

                uav.current_index += 1
                uav.last_event_time = self.time
                self.schedule(self.time + dt, "uav_arrival", uav.uav_id)

            elif event_type == "update_wp_risk":
                self.env.assign_random_risks()
                self.schedule(self.time + WP_RISK_UPDATE_INTERVAL, "update_wp_risk")

            elif event_type == "update_wp_revenue":
                self.env.assign_random_revenues()
                self.schedule(self.time + WP_REVENUE_UPDATE_INTERVAL, "update_wp_revenue")

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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    env = GridEnvironment(seed=42)
    env.assign_random_revenues()
    env.assign_random_risks()

    allocator = TaskAllocator(env)
    allocator.initialize_uavs()
    allocator.assign_random_sequences()

    for uav in allocator.uavs:
        print(f"UAV {uav.uav_id} sequence: {uav.sequence}")

    sim = Simulator(env, allocator.uavs)
    sim.run()