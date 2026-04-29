import heapq
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

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
    sequence: List[int] = field(default_factory=list)  # list of waypoint indices
    current_index: int = 0
    last_event_time: float = 0.0

    accumulated_risk: float = 0.0
    accumulated_revenue: float = 0.0
    remaining_flight_time: float = MAX_FLIGHT_TIME

    health: float = 1.0          # numerical: 1.0, 0.5, 0.0
    link_quality: float = 1.0    # in [0,1]
    collected_revenue: float = 0.0  # numerical: 0.0, 0.5, 1.0

    def current_wp_index(self) -> int:
        return self.sequence[self.current_index]

    def next_wp_index(self) -> Optional[int]:
        if self.current_index + 1 < len(self.sequence):
            return self.sequence[self.current_index + 1]
        return None

    # --- state update helpers (use environment waypoints) ---

    def update_accumulated_risk(self, env: "GridEnvironment", sim_time: float) -> None:
        idx = self.current_wp_index()
        wp = env.waypoints[idx]
        self.accumulated_risk += wp.risk
        if wp.risk == 1:
            print(f"[{sim_time:.1f}] UAV {self.uav_id} encountered risk at WP {idx}")

    def update_accumulated_revenue(self, env: "GridEnvironment", sim_time: float) -> None:
        idx = self.current_wp_index()
        wp = env.waypoints[idx]
        self.accumulated_revenue += wp.revenue
        if wp.revenue > 0:
            print(f"[{sim_time:.1f}] UAV {self.uav_id} collected revenue {wp.revenue:.1f} at WP {idx}")

    def update_health(self) -> None:
        # normalize used flight time
        flight_fraction = (MAX_FLIGHT_TIME - self.remaining_flight_time) / MAX_FLIGHT_TIME
        flight_fraction = max(0.0, min(1.0, flight_fraction))

        # normalize accumulated risk
        risk_fraction = min(1.0, self.accumulated_risk / MAX_WP_RISK)

        # linear combination
        score = (HEALTH_ALPHA * (1.0 - flight_fraction) +
                 HEALTH_BETA * (1.0 - risk_fraction))

        if score > HEALTH_GOOD_THRESHOLD:
            self.health = 1.0  # Good
        elif score > HEALTH_WARNING_THRESHOLD:
            self.health = 0.5  # Warning
        else:
            self.health = 0.0  # Critical

    def update_link_quality(self, env: "GridEnvironment", all_uavs: List["UAV"]) -> None:
        indicators: List[int] = []

        my_idx = self.current_wp_index()
        my_wp = env.waypoints[my_idx]

        for other in all_uavs:
            if other.uav_id == self.uav_id:
                continue
            other_wp = env.waypoints[other.current_wp_index()]
            dist = euclidean_distance(my_wp, other_wp)
            indicators.append(1 if dist <= COMMUNICATION_RANGE else 0)

        if indicators:
            self.link_quality = sum(indicators) / len(indicators)
        else:
            self.link_quality = 0.0

    def update_collected_revenue(self) -> None:
        max_possible_revenue = TOTAL_TARGETS * MAX_WP_REVENUE
        if max_possible_revenue <= 0:
            self.collected_revenue = 0.0
            return

        revenue_fraction = min(1.0, self.accumulated_revenue / max_possible_revenue)

        if revenue_fraction < REVENUE_LOW_THRESHOLD:
            self.collected_revenue = 0.0  # Low
        elif revenue_fraction < REVENUE_MEDIUM_THRESHOLD:
            self.collected_revenue = 0.5  # Medium
        else:
            self.collected_revenue = 1.0  # High

# ============================================================
# Environment
# ============================================================

class GridEnvironment:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.waypoints: List[Waypoint] = self._build_grid()
        self.depot_index: int = DEPOT_INDEX
        self.depot: Waypoint = self.waypoints[self.depot_index]
        self.target_waypoints: List[int] = self._select_random_targets()

    def _build_grid(self) -> List[Waypoint]:
        waypoints: List[Waypoint] = []
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                wp = Waypoint(x * GRID_SPACING, y * GRID_SPACING)
                waypoints.append(wp)
                # optional: comment out in final version if too verbose
                # print(f"Waypoint {len(waypoints)-1}: ({wp.x}, {wp.y}) with base revenue {wp.revenue}")
        return waypoints

    def _select_random_targets(self) -> List[int]:
        candidate_indices = [i for i in range(len(self.waypoints)) if i != self.depot_index]
        if TOTAL_TARGETS > len(candidate_indices):
            raise ValueError(
                f"TOTAL_TARGETS={TOTAL_TARGETS} exceeds available non-depot waypoints={len(candidate_indices)}"
            )
        return random.sample(candidate_indices, TOTAL_TARGETS)

    def assign_random_revenues(self) -> None:
        for idx in self.target_waypoints:
            self.waypoints[idx].revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)

    def assign_random_risks(self) -> None:
        for idx in self.target_waypoints:
            self.waypoints[idx].risk = random.choice([0, 1])

# ============================================================
# Utility functions
# ============================================================

def euclidean_distance(wp_a: Waypoint, wp_b: Waypoint) -> float:
    return math.hypot(wp_b.x - wp_a.x, wp_b.y - wp_a.y)

def travel_time(wp_a: Waypoint, wp_b: Waypoint) -> float:
    dist = euclidean_distance(wp_a, wp_b)
    return dist / UAV_SPEED

# ============================================================
# Task allocator
# ============================================================

class TaskAllocator:
    def __init__(self, environment: GridEnvironment):
        self.env = environment
        self.uavs: List[UAV] = []

    def initialize_uavs(self) -> None:
        for uid in range(1, TOTAL_UAVS + 1):
            uav = UAV(uav_id=uid)
            # start all UAVs at the depot index
            uav.sequence = [self.env.depot_index]
            self.uavs.append(uav)

    def assign_random_sequences(self) -> None:
        """
        Non-overlapping assignment: each target waypoint goes to exactly one UAV.
        """
        targets = list(self.env.target_waypoints)
        random.shuffle(targets)
        uav_idx = 0

        for wp_idx in targets:
            self.uavs[uav_idx].sequence.append(wp_idx)
            uav_idx = (uav_idx + 1) % len(self.uavs)

    def compute_sequence_revenue(self, sequence: List[int]) -> float:
        return sum(self.env.waypoints[idx].revenue for idx in sequence if idx != self.env.depot_index)

    def compute_sequence_risk(self, sequence: List[int]) -> float:
        return sum(self.env.waypoints[idx].risk for idx in sequence if idx != self.env.depot_index)

# ============================================================
# Backup policy (MDP-inspired)
# ============================================================

class BackupPolicy:
    """
    Simple parametric policy using health, link_quality, and collected_revenue.
    """

    @staticmethod
    def backup_score(uav: UAV) -> float:
        return (POLICY_WEIGHT_HEALTH * uav.health +
                POLICY_WEIGHT_LINK_QUALITY * uav.link_quality +
                POLICY_WEIGHT_COLLECTED_REVENUE * uav.collected_revenue)

    @staticmethod
    def decide_action(uav: UAV) -> str:
        """
        Returns one of: "return_to_depot", "backup", "continue".
        """
        # Always return if health is critical
        if uav.health == 0.0:
            return "return_to_depot"

        score = BackupPolicy.backup_score(uav)

        if score < BACKUP_SCORE_THRESHOLD:
            return "backup"

        return "continue"

# ============================================================
# Event simulation
# ============================================================

# global event list and counters (for simplicity)
events: List[Tuple[float, int, str, Optional[UAV]]] = []
event_counter = 0
sim_time = 0.0

def schedule(time: float, event_type: str, uav: Optional[UAV]) -> None:
    global event_counter
    event_counter += 1
    heapq.heappush(events, (time, event_counter, event_type, uav))

def schedule_initial_events(uavs: List[UAV]) -> None:
    for uav in uavs:
        schedule(0.0, "uav_arrival", uav)
    schedule(0.0, "update_wp_risk", None)
    schedule(0.0, "update_wp_revenue", None)

def run_events(env: GridEnvironment, uavs: List[UAV]) -> None:
    global sim_time
    while events and sim_time < SIM_TIME:
        sim_time, _, event_type, uav = heapq.heappop(events)

        if event_type == "uav_arrival" and uav is not None:
            idx = uav.current_wp_index()
            print(f"[{sim_time:.1f}] UAV {uav.uav_id} arrived at WP {idx}")

            travel_duration = sim_time - uav.last_event_time
            uav.remaining_flight_time -= travel_duration
            if uav.remaining_flight_time <= 0:
                print(f"[{sim_time:.1f}] UAV {uav.uav_id} ran out of flight time and crashed!")
                continue

            uav.last_event_time = sim_time

            uav.update_accumulated_risk(env, sim_time)
            uav.update_accumulated_revenue(env, sim_time)

            schedule(sim_time + UAV_HOVER_TIME, "uav_departure", uav)

        elif event_type == "uav_departure" and uav is not None:
            idx = uav.current_wp_index()
            print(f"[{sim_time:.1f}] UAV {uav.uav_id} departed from WP {idx}")

            hover_duration = sim_time - uav.last_event_time
            uav.remaining_flight_time -= hover_duration
            if uav.remaining_flight_time <= 0:
                print(f"[{sim_time:.1f}] UAV {uav.uav_id} ran out of flight time and crashed!")
                continue

            next_idx = uav.next_wp_index()
            if next_idx is None:
                print(f"[{sim_time:.1f}] UAV {uav.uav_id} finished sequence")
                continue

            # update local state variables
            uav.update_health()
            uav.update_link_quality(env, uavs)
            uav.update_collected_revenue()

            print(f"[{sim_time:.1f}] UAV {uav.uav_id} health: {uav.health}, "
                  f"link quality: {uav.link_quality*100:.1f}%, "
                  f"collected revenue: {uav.collected_revenue}")

            action = BackupPolicy.decide_action(uav)
            print(f"[{sim_time:.1f}] UAV {uav.uav_id} action: {action}")

            # For now, we only log the action and continue along the sequence.
            # Later you can implement special behavior for "backup" and "return_to_depot".
            curr_wp = env.waypoints[uav.current_wp_index()]
            next_wp = env.waypoints[next_idx]
            dt = travel_time(curr_wp, next_wp)

            uav.current_index += 1
            uav.last_event_time = sim_time

            schedule(sim_time + dt, "uav_arrival", uav)

        elif event_type == "update_wp_risk":
            env.assign_random_risks()
            schedule(sim_time + WP_RISK_UPDATE_INTERVAL, "update_wp_risk", None)

        elif event_type == "update_wp_revenue":
            env.assign_random_revenues()
            schedule(sim_time + WP_REVENUE_UPDATE_INTERVAL, "update_wp_revenue", None)

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

    # print assigned sequences for debugging
    for uav in allocator.uavs:
        print(f"UAV {uav.uav_id} sequence:", uav.sequence)

    schedule_initial_events(allocator.uavs)
    run_events(env, allocator.uavs)