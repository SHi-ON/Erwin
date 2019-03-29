import json
import numpy as np
import pandas as pd
# model
from keras.models import Sequential
from keras.layers.core import Dense
# Stochastic Gradient Descent optimizer
from keras.optimizers import sgd
import sys

class Catch(object):

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Output: new states and reward
        """
        samp_state = self.state

    def _draw_state(self):
        sample_state = self.state[0]
        # making a tuple of 2 grid_size-s for shape of canvas
        im_size = (self.grid_size,) * 2
        canvas = np.zeros(im_size)
        # array manipulation
        # [-1, ...] to get the bottom of the canvas
        canvas[sample_state[0], sample_state[1]] = 1  # draw fruit
        # TODO: basket is not drawn
        return canvas

    def observe(self):
        canvas = self._draw_state()
        # -1: reshape the remaining dimensions accordingly
        # means we need a row vector (doesn't matter how many columns it has)
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    # generates and adds a single random initial sample
    # (s, s', a)
    def reset(self):
        # return a 'size'-shaped array.
        # size = 0 or None -> returns a single value (not array)
        # size is redundant!
        n = np.random.randint(0, self.grid_size - 1, size=1)
        m = np.random.randint(1, self.grid_size - 2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]


class ExperienceReplay(object):
    """
    During gameplay all the experiences (s, a, r, s’) are stored in a replay memory.
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, max_memory=100, discount=0.9):
        """
        Setup
        max_memory: the maximum number of experiences we want to store.
        memory: a list of experiences.
        discount: the discount factor for future experience.

        In the memory the information whether the game ended at the state is stored separately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        env_dim = self.memory[0][0][0].shape[1]
        if env_dim != 18:
            print("DIMENSION is ", env_dim)
            sys.exit(-1)

        # We want to return an input and target vector with inputs from an observed state...
        # input features
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        # (min(len_memory, batch_size) <===> inputs.shape[0]
        # output action-values (Qs)
        targets = np.zeros((inputs.shape[0], num_actions))

        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i + 1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


def main():
    df = pd.read_csv("./datasets/project_sample1.csv")
    print(df.head())

    df = df.values
    # hyper-parameters
    epsilon = .1  # epsilon-greedy exploration
    grid_size = 3
    num_actions = 2  # [0, 10]
    actions_list = [0, 10]

    epoch = 10  # TODO: set properly

    max_memory = 500  # maximum number of experiences we are storing
    hidden_size = 100  # size of the hidden layers
    batch_size = 50  # number of experiences we use for training per batch TODO: try 1 as ipynb

    model = Sequential()
    # input_shape = grid_size * 2 = 18
    model.add(Dense(hidden_size, input_shape=(grid_size ** 2 * 2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    # lr: learning rate
    model.compile(sgd(lr=.2), "mse")

    model.summary()

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")

    # define environment/game
    # env = Catch(grid_size)

    # initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    win_cnt = 0  # reset the win counter

    # FIXME: added from ipynb
    win_hist = []  # keep track of win count history

    for e in range(epoch):
        loss = 0.
        # env.reset()
        game_over = False
        # get initial input
        # input_t = env.observe()
        row = e * 200
        input_t = df[row, 1:19].reshape((1, -1))

        for i in range(200):
            input_tm1 = input_t
            # get next action

            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            # input_t, reward, game_over = env.act(action)
            # if reward == 1:
            #     win_cnt += 1
            input_t = df[row + i, 1:19].reshape((1, -1))
            reward = df[row + i, 20]
            game_over = e == 200 - 1

            # store experience
            # ([s, a, r, s'], game_over)
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # loss += model.train_on_batch(inputs, targets)[0]
            loss += model.train_on_batch(inputs, targets)

            print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("./models/model.h5", overwrite=True)
    with open("./models/model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)


if __name__ == "__main__":
    main()
