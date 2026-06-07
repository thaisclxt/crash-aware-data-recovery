import math

def euclidean_distance(origin, destination):
    return math.hypot(destination[0] - origin[0], destination[1] - origin[1])


def travel_time(origin, destination, speed):
    return euclidean_distance(origin, destination) / speed


def prepare_mission(sequence) -> None:
    if not sequence:
        return
    
    m_j = 0

    depot = (0,0)
    prep = 1
    hover = 5
    maxft = 20
    speed = 10
    
    x = travel_time(
        origin=depot,
        destination=sequence[0],
        speed=speed,
    ) + prep

    y = travel_time(
        origin=sequence[-1],
        destination=depot,
        speed=speed,
    )

    k = travel_time(
        origin=sequence[-1],
        destination=sequence[0],
        speed=speed
    )

    z = hover
    for wp_id, nxt_wp_id in zip(sequence, sequence[1:]):
        z += travel_time(
            origin=wp_id,
            destination=nxt_wp_id,
            speed=speed
        ) + hover

    while True:
        if (x + y + ((m_j+1) * z) + (m_j * k)) > maxft:
            break

        time = x + y + ((m_j+1) * z) + (m_j * k)
        m_j += 1
        

    print(m_j)
    print(time)
        

prepare_mission([(5,0)])