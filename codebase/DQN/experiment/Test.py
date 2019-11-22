# json for loading and saving the model (optional)
import json
# matplotlib for rendering
import matplotlib.pyplot as plt
# numpy for handling matrix operations
import numpy as np
import pandas as pd
from keras.models import model_from_json

# time, to, well... keep track of time TODO: use it somewhere!
# Python image libarary for rendering
# IPython display for making sure we can render the frames
# seaborn for rendering
import seaborn

# Setup matplotlib so that it runs nicely in iPython
# %matplotlib inline
# setting up seaborn
seaborn.set()

if __name__ == "__main__":
    # Make sure this grid size matches the value used for training
    grid_size = 3
    v1_max = 200
    epoch = 10

    df = pd.read_csv("./dataset/project_sample1.csv")
    print(df.head())

    df = df.values

    with open("./models/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("./models/model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    # env = Catch(grid_size)
    c = 0
    for e in range(epoch):
        loss = 0.
        # env.reset()f
        game_over = False
        # get initial input
        # input_t = env.observe()
        row = e * 200
        input_t = df[row, 1:19].reshape((1, -1))

        # plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
        plt.imshow(input_t, interpolation='none', cmap='gray')
        plt.show()
        plt.savefig("./figs/%03d.png" % c)
        c += 1
        for i in range(v1_max):
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            # input_t, reward, game_over = env.act(action)
            input_t = df[row + i, 1:19].reshape((1, -1))
            reward = df[row + i, 20]
            game_over = e == v1_max - 1

            # plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
            plt.imshow(input_t, interpolation='none', cmap='gray')
            plt.show()
            plt.savefig("./figs/%03d.png" % c)
            c += 1
