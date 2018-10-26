import numpy as np

# initiliaze grid parameters
state_action_values = np.full((4, 8 , 8), -100)

# basic characteristics of problem
gamma = 1
alpha = 0.2
epsilon = 0.3
episodes = 1000

def check_position(x,y):

    # check for walls
    if x == 1:
        if y == 2 or y == 3 or y == 4 or y == 5:
            return 1
    if x == 2:
        if y == 5:
            return 1
    if x == 3:
        if y == 5:
            return 1
    if x == 4:
        if y == 5:
            return 1
    if x == 6:
        if y == 1 or y == 2 or y == 3 or y == 4:
            return 1

    # check for off-grid
    if x < 0 or x > 7 or y < 0 or y > 7:
        return 1

    # check for snake pits
    if x == 5:
        if y == 4:
            return 2

    # check for treasure
    if x == 7:
        if y == 7:
            return 3

    return 0

for epis in range(episodes):

    # pseudo-random initilization of starting position of agent
    check = 1

    while check > 0:
        [x,y] = np.random.randint(low = 0, high = 8, size = (2, 1), dtype = 'l')
        [x,y] = [x[0], y[0]]
        check = check_position(x, y)

    while check < 2:
        # run an eps-greedy policy
        eps = np.random.uniform(low = 0, high = 1)

        if eps < 0.1:
            x_step = -1
            y_step = -1

            # restrict directions as to not allow diagonal walking
            while x_step + y_step == 0 or x_step == y_step:

                # generate random x direction
                x_step = np.random.randint(low = -1, high = 2, dtype='l')
                y_step = np.random.randint(low = -1, high = 2, dtype='l')

            # define steps
            if x_step == 0 and y_step == 1:
                # up
                step = 0
            if x_step == 0 and y_step == -1:
                # down
                step = 1
            if x_step == 1 and y_step == 0:
                # right
                step = 2
            if x_step == -1 and y_step == 0:
                # left
                step = 3

            check = check_position(x + x_step, y + y_step)

            if check == 1:
                state_action_values[step, x, y] = -1
            if check == 2:
                state_action_values[step, x, y] = -20
                break
            if check == 3:
                state_action_values[step, x, y] = 10
                break
            if check == 0:
                state_action_values[step, x, y] = state_action_values[step, x, y] +\
                    alpha * (-1 + gamma * max(state_action_values[:, x + x_step, y + y_step]) - state_action_values[step, x, y])
                x = x + x_step
                y = y + y_step
        else:
            # take greendy action
            step = np.argmax(state_action_values[:, x, y])

            # define steps
            if step == 0:
                # up
                x_step = 0
                y_step = 1
            if step == 1:
                # down
                x_step = 0
                y_step = -1
            if step == 2:
                # right
                x_step = 1
                y_step = 0
            if step == 3:
                # left
                x_step = -1
                y_step = 0

            check = check_position(x + x_step, y + y_step)

            if check == 1:
                state_action_values[step, x, y] = -1
            if check == 2:
                state_action_values[step, x, y] = -20
                break
            if check == 3:
                state_action_values[step, x, y] = 10
                break
            if check == 0:
                state_action_values[step, x, y] = state_action_values[step, x, y] +\
                    alpha * (-1 + gamma * max(state_action_values[:, x + x_step, y + y_step]) - state_action_values[step, x, y])
                x = x + x_step
                y = y + y_step


test = np.zeros(shape = (8, 8))
for i in range(8):
    for j in range(8):
        test[i,j] = np.argmax(state_action_values[:,i,j])



print(test)