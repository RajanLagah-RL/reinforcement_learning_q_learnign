import gym
import numpy as np

env = gym.make("MountainCar-v0")

done = False
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 100
# this print B(2,) which mean it is 2d matrix of real numbers
print(env.observation_space)

#this will return max in each dimension
print(env.observation_space.high)

#this will return min in each dimension
print(env.observation_space.low)

# epsilon is for randomness as without it if program find success way then it will concentrate on it only amd
# will not find other gud way 
epsilon = 0.5


START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


# we will try to construct matrix to accamodate many values q-table
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

#TO find points at eqal distance
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Random matrix having values for all/ max possible actions
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

prev_state = env.reset()
discrete_state = get_discrete_state(prev_state)
# values of action at home
print("prev home action:")
print(q_table[discrete_state])

for episode in range(EPISODES):

    if  episode % SHOW_EVERY  == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)

        action = np.argmax(q_table[discrete_state])
        new_state,reward,done,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+(action,)]
            new_q = (1 - LEARNING_RATE)* current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state+(action,)] = 0

        discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
env.close()

discrete_state = get_discrete_state(env.reset())
print("after home action:")
print(q_table[discrete_state])
