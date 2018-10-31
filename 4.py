# import necessary module
import numpy as np
import matplotlib.pyplot as plt
import random

# initialize matrix to store the state-action values
state_action_values_Q = np.full((4, 8, 8), 0, dtype="float")
state_action_values_SARSA = np.full((4, 8, 8), 0, dtype="float")

# define parameters of problem solution
gamma = 1
alpha = 0.4
epsilon = 0.006
episodes = 1000
avg_reward = np.zeros((episodes, 14), dtype="float")


def check_position(x, y):
    """"This function checks whether the proposed position (determined with x
     and y coordinates) is on the grid, terminal or a wall."""

    # check if position is a wall
    if y == 1 and x in {2, 3, 4, 5}:
        return 1
    if x == 5 and y in {2, 3, 4}:
        return 1
    if y == 6 and x in {1, 2, 3}:
        return 1

    # check if position is off-grid
    if any(np.array([x, y]) > 7) or any(np.array([x, y]) < 0):
        return 1

    # check if position is a snake pit
    if y == 5 and x == 4:
        return 2

    # check if position is a treasure
    if y == 7 and x == 7:
        return 3

    # if position is regular return 0
    return 0


# run episodes to get estimates for the state-action values of Q-value
for epis in range(episodes):

    # pseudo-random initialization of starting position of agent
    check = 1
    while check > 0:
        [x, y] = np.random.randint(low=0, high=8, size=(2, 1), dtype="l")
        [x, y] = [x[0], y[0]]
        check = check_position(x, y)

    # agent is allowed to move around until caught in a terminal state
    while check < 2:

        # run an epsilon-greedy policy
        eps = np.random.uniform(low=0, high=1)

        # explore
        if eps < epsilon:

            # values to initialize the while loop for random step generation
            x_step = -1
            y_step = -1

            # restrict directions as to not allow diagonal walking
            while x_step + y_step == 0 or x_step == y_step:
                # generate random manhattan move
                x_step = np.random.randint(low=-1, high=2, dtype="l")
                y_step = np.random.randint(low=-1, high=2, dtype="l")

            # convert the step into integer code for direction for matrix index
            if x_step == 0 and y_step == -1:
                # up
                step = 0
            elif x_step == 0 and y_step == 1:
                # down
                step = 1
            elif x_step == 1 and y_step == 0:
                # right
                step = 2
            else:
                # left
                step = 3

            # check to what kind of state the proposed step leads
            check = check_position(x + x_step, y + y_step)

            if check == 1:
                # wall
                state_action_values_Q[step, y, x] += alpha * (-1 +
                                                              gamma * max(
                            state_action_values_Q[:, y, x]) -
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += -1
            elif check == 2:
                # snake pit terminal state
                state_action_values_Q[step, y, x] += alpha * (-20 - \
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += -20
                break
            elif check == 3:
                # treasure terminal state
                state_action_values_Q[step, y, x] += alpha * (10 - \
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += 10
                break
            else:
                # normal state

                # Q-learning algorithm-----------------------------------------
                state_action_values_Q[step, y, x] += alpha * (-1 +
                                                              gamma * max(
                            state_action_values_Q[:, y + y_step, x + x_step]) -
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += -1
                avg_reward[epis, 1] += 1

                # update new location
                x += x_step
                y += y_step

        # exploit
        else:
            # take the greedy action
            best_actions = np.where(state_action_values_Q[:, y, x] == \
                                    max(state_action_values_Q[:, y, x]))[0]
            step = random.choice(best_actions)

            # define steps
            if step == 0:
                # up
                x_step, y_step = 0, -1
            if step == 1:
                # down
                x_step, y_step = 0, 1
            if step == 2:
                # right
                x_step, y_step = 1, 0
            if step == 3:
                # left
                x_step, y_step = -1, 0

            # check to what kind of state the proposed step leads
            check = check_position(x + x_step, y + y_step)

            if check == 1:
                # wall
                state_action_values_Q[step, y, x] += alpha * (-1 +
                                                              gamma * max(
                            state_action_values_Q[:, y, x]) -
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += -1
            elif check == 2:
                # snake pit terminal state
                state_action_values_Q[step, y, x] += alpha * (-20 - \
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += -20
                break
            elif check == 3:
                # treasure terminal state
                state_action_values_Q[step, y, x] += alpha * (10 - \
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += 10
                break
            else:
                # normal state

                # Q-learning algorithm-----------------------------------------
                state_action_values_Q[step, y, x] += alpha * (-1 +
                                                              gamma * max(
                            state_action_values_Q[:, y + y_step, x + x_step]) -
                                                              state_action_values_Q[
                                                                  step, y, x])
                avg_reward[epis, 0] += -1

                # update new location
                x += x_step
                y += y_step

# run episodes to get estimates for the state-action values
for epis in range(episodes):

    # pseudo-random initialization of starting position of agent
    check = 1
    while check > 0:
        [x, y] = np.random.randint(low=0, high=8, size=(2, 1), dtype="l")
        [x, y] = [x[0], y[0]]
        check = check_position(x, y)

    # agent is allowed to move around until caught in a terminal state
    while check < 2:

        # run an epsilon-greedy policy
        eps = np.random.uniform(low=0, high=1)

        # explore
        if eps < epsilon:

            # values to initialize the while loop for random step generation
            x_step = -1
            y_step = -1

            # restrict directions as to not allow diagonal walking
            while x_step + y_step == 0 or x_step == y_step:
                # generate random manhattan move
                x_step = np.random.randint(low=-1, high=2, dtype="l")
                y_step = np.random.randint(low=-1, high=2, dtype="l")

            # convert the step into integer code for direction for matrix index
            if x_step == 0 and y_step == -1:
                # up
                step = 0
            elif x_step == 0 and y_step == 1:
                # down
                step = 1
            elif x_step == 1 and y_step == 0:
                # right
                step = 2
            else:
                # left
                step = 3

            # check to what kind of state the proposed step leads
            check = check_position(x + x_step, y + y_step)

            if check == 1:
                # wall
                state_action_values_SARSA[step, y, x] += alpha * (-1 +
                                                                  gamma * sum(
                            state_action_values_SARSA[:, y, x]) / 4 - \
                                                                  state_action_values_SARSA[
                                                                      step, y, x])
                avg_reward[epis, 1] += -1
            elif check == 2:
                # snake pit terminal state
                state_action_values_SARSA[step, y, x] += alpha * (-20 - \
                                                                  state_action_values_SARSA[
                                                                      step, y, x])
                avg_reward[epis, 1] += -20
                break
            elif check == 3:
                # treasure terminal state
                state_action_values_SARSA[step, y, x] = alpha * (10 - \
                                                                 state_action_values_SARSA[
                                                                     step, y, x])
                avg_reward[epis, 1] += 10
                break
            else:
                # normal state

                # Expected value SARSA algorithm--------------------------------
                state_action_values_SARSA[step, y, x] += alpha * (-1 +
                                                                  gamma * sum(
                            state_action_values_SARSA[:, y + y_step, x + \
                                                                     x_step]) / 4 -
                                                                  state_action_values_SARSA[
                                                                      step, y, x])
                avg_reward[epis, 1] += -1

                # update new location
                x += x_step
                y += y_step

        # exploit
        else:
            # take the greedy action
            best_actions = np.where(state_action_values_SARSA[:, y, x] == \
                                    max(state_action_values_SARSA[:, y, x]))[0]
            step = random.choice(best_actions)

            # define steps
            if step == 0:
                # up
                x_step, y_step = 0, -1
            if step == 1:
                # down
                x_step, y_step = 0, 1
            if step == 2:
                # right
                x_step, y_step = 1, 0
            if step == 3:
                # left
                x_step, y_step = -1, 0

            # check to what kind of state the proposed step leads
            check = check_position(x + x_step, y + y_step)

            if check == 1:
                # wall
                state_action_values_SARSA[step, y, x] += alpha * (-1 +
                                                                  gamma * sum(
                            state_action_values_SARSA[:, y, x]) / 4 - \
                                                                  state_action_values_SARSA[
                                                                      step, y, x])
                avg_reward[epis, 1] += -1
            elif check == 2:
                # snake pit terminal state
                state_action_values_SARSA[step, y, x] = alpha * (-20 - \
                                                                 state_action_values_SARSA[
                                                                     step, y, x])
                avg_reward[epis, 1] += -20
                break
            elif check == 3:
                # treasure terminal state
                state_action_values_SARSA[step, y, x] = alpha * (10 - \
                                                                 state_action_values_SARSA[
                                                                     step, y, x])
                avg_reward[epis, 1] += 10
                break
            else:
                # normal state

                # Expected value SARSA algorithm--------------------------------
                state_action_values_SARSA[step, y, x] += alpha * (-1 +
                                                                  gamma * sum(
                            state_action_values_SARSA[:, y + y_step, x + \
                                                                     x_step]) / 4 -
                                                                  state_action_values_SARSA[
                                                                      step, y, x])
                avg_reward[epis, 1] += -1

                # update new location
                x += x_step
                y += y_step

# create arrays to store the optimal policy steps in
policy_Q = np.zeros(shape=(8, 8), dtype="str")
policy_SARSA = np.zeros(shape=(8, 8), dtype="str")

# dictionary for the step names
direction = {0: 'up', 1: 'down', 2: 'right', 3: 'left'}

# translate the number coding to string policies
for i in range(8):
    for j in range(8):
        policy_Q[i, j] = (direction[np.argmax(state_action_values_Q[:,
                                              i, j])])
        policy_SARSA[i, j] = (direction[np.argmax(state_action_values_SARSA[:,
                                                  i, j])])

# visualize walls, snake pit and treasure with a X
policy_Q[1, 2:6] = "X"
policy_Q[2:5, 5] = "X"
policy_Q[5, 4] = "X"
policy_Q[6, 1:4] = "X"
policy_Q[7, 7] = "X"

policy_SARSA[1, 2:6] = "X"
policy_SARSA[2:5, 5] = "X"
policy_SARSA[5, 4] = "X"
policy_SARSA[6, 1:4] = "X"
policy_SARSA[7, 7] = "X"

print(policy_SARSA)
print(policy_Q)

fig1 = plt.figure(1, figsize=(10, 6))
x_range = range(episodes)

# generating weights for polynomial function with degree =15
weights_Q = np.polyfit(x_range, avg_reward[:, 0], 15)
weights_SARSA = np.polyfit(x_range, avg_reward[:, 1], 15)

# generating model with the given weights
model_Q = np.poly1d(weights_Q)
model_SARSA = np.poly1d(weights_SARSA)

pred_plot_Q = model_Q(x_range)
pred_plot_SARSA = model_SARSA(x_range)
plt.plot(x_range, pred_plot_Q)
plt.plot(x_range, pred_plot_SARSA)

plt.title('Reward for different episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(["Q-learning", "Sarsa"])