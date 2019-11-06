import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
# Stochastic Gradient Descent optimizer
# from keras.optimizers import sgd
from keras.optimizers import rmsprop
# from keras.optimizers import adam

import matplotlib.pyplot as plt

# sampling handler class and parameters
from util import *
import sample as samp
# Experience Replay adapter class
import memory as memo


# import os
# os.chdir("./codebase/DQN")


def network_builder():
    model = Sequential()
    # grid_size x 2 = 18 inputs
    model.add(Dense(hidden_size, input_shape=(grid_size * 2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    # num_actions = 2 outputs
    model.add(Dense(num_actions))
    # lr: learning rate
    model.compile(rmsprop(lr=alpha), "mse")

    model.summary()
    return model


def train(df):

    model = network_builder()

    exp_rep = memo.ExperienceReplay(mem_len=mem_len, disc=gamma)
    sample = samp.Sample(df, v1_max, reward_col)

    epochs = sample.get_epochs()
    loss_df = pd.DataFrame({'step': range(1, v1_max * epochs + 1), 'loss': range(1, v1_max * epochs + 1)})
    for ep in range(epochs):
        loss = 0.
        next_state = sample.get_init_sample(ep)

        # duplicate state prevention in initial step w/ start from 1
        for i in range(1, v1_max):
            state = next_state

            q = model.predict(state)
            action = np.argmax(q[0])
            # print("Action in ", ep, "and step", i)

            reward = sample.get_reward(ep, i)

            next_state = sample.get_sample(ep, i)

            game_over = sample.is_over(ep)

            # store experience
            # ([s, a, r, s'], game_over)
            exp_rep.store([state, action, reward, next_state], game_over)

            # retrieve proper features and targets
            inputs, targets = exp_rep.get_batch(model, batch_size=batch_size)

            # loss += model.train_on_batch(inputs, targets)[0]
            loss += model.train_on_batch(inputs, targets)

            print("before store: ", str(ep * v1_max + i), loss)
            plotter(loss_df, ep * v1_max + i, loss)

    plt.plot('loss', 'step', data=loss_df, marker='o')
    plt.ylabel('Loss over episodes of training')
    plt.show()
        # print("Epoch {:4d}/{:4d} | Loss {:.4f}".format(ep, epochs, loss))
    model_export(model)


if __name__ == '__main__':
    data_frame = dataset_import()

    a_time = time.time()
    train(data_frame)
    z_time = time.time()

    stopwatch_log(a_time, z_time)
    print("Training is finished")


