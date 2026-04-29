import random
import numpy as np

HEALTH = {0: "Normal", 1: "Alert", 2: "Critical"}
LINK = {0: "Good", 1: "Fair", 2: "Poor"}
DATA_VAL = {0: "Low", 1: "High"}
RISK = {0: "Low", 1: "High"}

def MDP_state():
    H = random.choice(list(HEALTH.keys()))
    Q = random.choice(list(LINK.keys()))
    D = random.choice(list(DATA_VAL.keys()))
    R = random.choice(list(RISK.keys()))
    return (H, Q, D, R)

def crash_probability(state):
    H, Q, D, R = state

    # Base on health
    if H == 0:   # Normal
        p = 0.1
    elif H == 1: # Alert
        p = 0.4
    else:        # Critical
        p = 0.8

    # Add extra risk if R == High
    if R == 1:
        p += 0.15

    # Worse link -> slightly higher risk
    if Q == 1:   # Fair
        p += 0.05
    elif Q == 2: # Poor
        p += 0.1

    return max(0.0, min(1.0, p))

def fixed_threshold_policy(state, theta=0.5):
    p_crash = crash_probability(state)
    if p_crash >= theta:
        return 1  # backup
    else:
        return 0  # no_send

# ---- Environment step: simulate crash, backup, and reward ----
def step(state, action):
    H, Q, D, R = state
    p_crash = crash_probability(state)
    crashed = random.random() < p_crash

    # Communication cost (only if backup)
    comm_cost = 0.05 if action == 1 else 0.0

    reward = 0.0

    if D == 1:  # High-value data
        if crashed and action == 0:
            reward -= 5.0   # lost critical data
        elif crashed and action == 1:
            reward += 3.0   # data preserved
        else:
            reward += 0.5   # mission continues with critical data safe so far
    else:       # Low-value data
        if crashed and action == 0:
            reward -= 0.5   # minor loss
        elif crashed and action == 1:
            reward += 0.2   # data preserved but not so important
        else:
            reward += 0.1   # no crash, low data

    # Subtract communication cost
    reward -= comm_cost

    next_state = MDP_state()
    done = True
    info = {"crashed": crashed, "p_crash": p_crash}

    return next_state, reward, done, info

def run_simulation(num_episodes=10000, theta=0.5):
    total_reward = 0.0
    crashes = 0
    backups = 0
    lost_critical = 0

    for _ in range(num_episodes):
        state = MDP_state()
        action = fixed_threshold_policy(state, theta=theta)
        if action == 1:
            backups += 1
        _, reward, _, info = step(state, action)
        total_reward += reward
        if info["crashed"]:
            crashes += 1
            # count critical data losses
            H, Q, D, R = state
            if D == 1 and action == 0:
                lost_critical += 1

    avg_reward = total_reward / num_episodes
    crash_rate = crashes / num_episodes
    backup_rate = backups / num_episodes
    lost_critical_rate = lost_critical / num_episodes

    return {
        "avg_reward": avg_reward,
        "crash_rate": crash_rate,
        "backup_rate": backup_rate,
        "lost_critical_rate": lost_critical_rate
    }

if __name__ == "__main__":
    for theta in [0.3, 0.5, 0.7]:
        stats = run_simulation(num_episodes=5000, theta=theta)
        print(f"Theta={theta}: {stats}")
