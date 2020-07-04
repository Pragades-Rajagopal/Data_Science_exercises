import numpy as np

gamma = 0.75
alpha = 0.9

loc_to_state = {
    'L1' : 0,
    'L2' : 1,
    'L3' : 2,
    'L4' : 3,
    'L5' : 4,
    'L6' : 5,
    'L7' : 6,
    'L8' : 7,
    'L9' : 8
}

actions = [0,1,2,3,4,5,6,7,8]

rewards = np.matrix([
    [0,1,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,1,0],
    [0,0,0,0,1,0,1,0,1],
    [0,0,0,0,0,0,0,1,0]
])

# Mapping indices to Locations
state_to_loc = dict((state,location) for location, state in loc_to_state.items())

def optimal_route(start_location, end_location):
    new_rewards = np.copy(rewards)
    ending_state = loc_to_state[end_location]
    new_rewards[ending_state, ending_state] = 999

    Q = np.array(np.zeros([9,9]))
    
    for i in range(1000):
        current_state = np.random.randint(0,9)
        playable_action = []

        for j in range(9):
            if new_rewards[current_state, j] > 0:
                playable_action.append(j)

        next_state = np.random.choice(playable_action)

    # Temporal difference
        TD = new_rewards[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]

        Q[current_state, next_state] += alpha * TD

    route = [start_location]
    next_location = start_location

    while(next_location != end_location):
        starting_state = loc_to_state[start_location]

        next_state = np.argmax(Q[starting_state,])

        next_location = state_to_loc[next_state]
        route.append(next_location)

        start_location = next_location

    return route

print(optimal_route('L6', 'L1'))


