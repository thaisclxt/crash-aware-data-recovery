import heapq, random, math

GRID_SIZE = 13
WAYPOINT_SPACING = 50.0
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
HEALTH_BETHA = 0.4
HEALTH_GOOD_THRESHOLD = 0.7
HEALTH_WARNING_THRESHOLD = 0.4

REVENUE_LOW_THRESHOLD = 0.4
REVENUE_MEDIUM_THRESHOLD = 0.7

POLICY_WEIGHT_HEALTH = 0.5
POLICY_WEIGHT_LINK_QUALITY = 0.3
POLICY_WEIGHT_COLLECTED_REVENUE = 0.2

BACKUP_SCORE_THRESHOLD = 0.5

time = 0.0
events = []  # (time, event_counter, event_type, uav)
event_counter = 0

def schedule(time, event_type, uav=None):
    global event_counter
    event_counter += 1
    heapq.heappush(events, (time, event_counter, event_type, uav))

class Waypoint:
    def __init__(self, wid, x, y):
        self.id = wid
        self.x = x
        self.y = y
        self.risk = 0
        self.revenue = 0

    def update_risk(self):
        self.risk = random.choice([0, 1]) # (0 for safe, 1 for risky)

    def update_revenue(self):
        self.revenue = random.uniform(MIN_WP_REVENUE, MAX_WP_REVENUE)

class UAV:
    def __init__(self, uid, route):
        self.id = uid
        self.route = route
        self.current_index = 0
        self.last_event_time = 0.0

        self.accumulated_risk = 0
        self.accumulated_revenue = 0
        self.remaining_flight_time = MAX_FLIGHT_TIME

        self.x = route[0].x
        self.y = route[0].y

        self.health = 0
        self.link_quality = 0
        self.collected_revenue = 0

    @property
    def current_wp(self):
        return self.route[self.current_index]

    def next_wp(self):
        if self.current_index + 1 < len(self.route):
            return self.route[self.current_index + 1]
        return None
    
    def update_accumulated_risk(self):
        self.accumulated_risk += self.current_wp.risk
        if self.current_wp.risk == 1:
            print(f"[{time:.1f}] UAV {self.id} encountered risk at WP {self.current_wp.id}")

    def update_accumulated_revenue(self):
        self.accumulated_revenue += self.current_wp.revenue
        if self.current_wp.revenue > 0:
            print(f"[{time:.1f}] UAV {self.id} collected revenue {self.current_wp.revenue:.1f} at WP {self.current_wp.id}")

    # based on the accumulated risk since last depot visit and remaining flight time since last depot visit
    def health_state(self):
        # normalize flight time
        fligth_fraction = (MAX_FLIGHT_TIME - self.remaining_flight_time) / MAX_FLIGHT_TIME
        fligth_fraction = max(0.0, min(1.0, fligth_fraction))  # clamp to [0, 1]

        # normalize accumulated risk
        risk_fraction = min(1.0, self.accumulated_risk / MAX_WP_RISK)

        # linear combination of remaining flight time and accumulated risk
        alpha = HEALTH_ALPHA # weight for remaining flight time
        betha = HEALTH_BETHA # weight for accumulated risk

        # higher score means better health: 1.0 = perfect, 0.0 = worst
        score = alpha * (1.0 - fligth_fraction) + betha * (1.0 - risk_fraction)

        if score > HEALTH_GOOD_THRESHOLD:
            self.health = 1.0  # Good
        elif score > HEALTH_WARNING_THRESHOLD:
            self.health = 0.5  # Warning
        else:
            self.health = 0.0  # Critical
            
    # based on the percentage of UAVs with a good reception coverage, that means the closest ones at the current waypoint    
    def link_quality_state(self, all_uavs):
        indicators = []

        for other in all_uavs:
            if other.id == self.id:
                continue
            dist = distance(self.current_wp, other.current_wp)

            # good link quality (1) if within communication range, otherwise bad (0)
            indicators.append(1 if dist <= COMMUNICATION_RANGE else 0)

        # calculate the percentage of UAVs with good link quality
        if indicators:
            self.link_quality = sum(indicators) / len(indicators)
        else:
            self.link_quality = 0.0  # no other UAVs, assume no link quality

    def collected_revenue_state(self):
        # normalize collected revenue
        max_possible_revenue = TOTAL_TARGETS * MAX_WP_REVENUE
        revenue_fraction = min(1.0, self.accumulated_revenue / max_possible_revenue)

        if revenue_fraction < REVENUE_LOW_THRESHOLD:
            self.collected_revenue = 0.0  # Low
        elif revenue_fraction < REVENUE_MEDIUM_THRESHOLD:
            self.collected_revenue = 0.5  # Medium
        else:
            self.collected_revenue = 1.0  # High

    def should_backup(self):
        # calculate a score based on health, link quality, and collected revenue
        score = (POLICY_WEIGHT_HEALTH * self.health +
                 POLICY_WEIGHT_LINK_QUALITY * self.link_quality +
                 POLICY_WEIGHT_COLLECTED_REVENUE * self.collected_revenue)
        
        if self.health == 0.0:  # Critical health
            return True

        return score < BACKUP_SCORE_THRESHOLD

def build_grid():
    waypoints = []
    wid = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = i * WAYPOINT_SPACING
            y = j * WAYPOINT_SPACING
            waypoints.append(Waypoint(wid, x, y))
            wid += 1

            # print(f"Waypoint {wid}: ({x}, {y})")
    return waypoints

def distance(wp1, wp2):
    dx = wp1.x - wp2.x
    dy = wp1.y - wp2.y
    return math.hypot(dx, dy)

def travel_time(wp1, wp2):
    return distance(wp1, wp2) / UAV_SPEED

# assign fixed routes to 3 UAVs (for testing)
def assign_fixed_routes(depot, waypoints):
    route1 = [depot, waypoints[165], waypoints[166], waypoints[167], waypoints[168]]
    route2 = [depot, waypoints[14], waypoints[15], waypoints[16], waypoints[17]]
    route3 = [depot, waypoints[27], waypoints[28], waypoints[29], waypoints[30]]

    uav1 = UAV(uid=1, route=route1)
    uav2 = UAV(uid=2, route=route2)
    uav3 = UAV(uid=3, route=route3)

    return [uav1, uav2, uav3]

def assign_routes():
    while target_waypoints:
        for uav in uavs:
            if not target_waypoints:
                break
            # select random
            target_index = target_waypoints.pop()
            target_wp = waypoints[target_index]
            uav.route.append(target_wp)

# build environment
waypoints = build_grid()
depot = waypoints[0]

# global pool of target waypoints (excluding depot)
target_waypoints = random.sample(range(1, len(waypoints)), TOTAL_TARGETS)

# assign revenue and risk to each target waypoint
for wp in waypoints[1:]:  # exclude depot
    wp.update_risk()
    wp.update_revenue()

uavs = []

# initialize UAVs at the depot with empty routes
for uid in range(1, TOTAL_UAVS + 1):
    uav = UAV(uid=uid, route=[depot])
    uavs.append(uav)

# only for testing
# uavs = assign_fixed_routes(depot, waypoints)

assign_routes()

# print the assigned routes for each UAV
for uav in uavs:
    wp_ids = [wp.id for wp in uav.route]
    print(f"UAV {uav.id} route:", wp_ids)

# initial event: each UAV starts at its first waypoint at time 0
for uav in uavs:
    schedule(0.0, "uav_arrival", uav)

schedule(0.0, "update_wp_risk", None)
schedule(0.0, "update_wp_revenue", None)

# event loop
while events and time < SIM_TIME:
    time, _, event_type, uav = heapq.heappop(events)

    if event_type == "uav_arrival":
        print(f"[{time:.1f}] UAV {uav.id} arrived at WP {uav.current_wp.id}")

        travelling_time = time - uav.last_event_time

        uav.remaining_flight_time = uav.remaining_flight_time - travelling_time
        if uav.remaining_flight_time <= 0:
            print(f"[{time:.1f}] UAV {uav.id} ran out of flight time and crashed!")
            continue

        uav.last_event_time = time

        uav.update_accumulated_risk()
        uav.update_accumulated_revenue()

        # schedule departure after hover time
        dt = UAV_HOVER_TIME
        schedule(time + dt, "uav_departure", uav)

    elif event_type == "uav_departure":
        print(f"[{time:.1f}] UAV {uav.id} departed from WP {uav.current_wp.id}")

        hovering_time = time - uav.last_event_time

        uav.remaining_flight_time = uav.remaining_flight_time - hovering_time
        if uav.remaining_flight_time <= 0:
            print(f"[{time:.1f}] UAV {uav.id} ran out of flight time and crashed!")
            continue
        
        nxt = uav.next_wp()
        if nxt is None:
            print(f"[{time:.1f}] UAV {uav.id} finished route")
            continue
        
        curr = uav.current_wp
        dt = travel_time(curr, nxt)

        uav.health_state()
        uav.link_quality_state(uavs)
        uav.collected_revenue_state()

        print(f"[{time:.1f}] UAV {uav.id} health state: {uav.health}, link quality: {uav.link_quality*100:.1f}%, collected revenue state: {uav.collected_revenue}")

        if uav.should_backup():
            print(f"[{time:.1f}] UAV {uav.id} decided to BACKUP (return to depot)")

        uav.current_index += 1
        uav.x, uav.y = nxt.x, nxt.y

        uav.last_event_time = time

        # schedule arrival at next waypoint (no hover)
        schedule(time + dt, "uav_arrival", uav)

    # update risk of all waypoints at regular intervals
    elif event_type == "update_wp_risk":
        for wp in waypoints[1:]:
            wp.update_risk()
        schedule(time + WP_RISK_UPDATE_INTERVAL, "update_wp_risk", None)

    # update revenue of all waypoints at regular intervals
    elif event_type == "update_wp_revenue":
        for wp in waypoints[1:]:
            wp.update_revenue()
        schedule(time + WP_REVENUE_UPDATE_INTERVAL, "update_wp_revenue", None)
