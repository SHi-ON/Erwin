
# json for loading and saving the model (optional)
import json
# matplotlib for rendering
import matplotlib.pyplot as plt
# numpy for handling matrix operations
import numpy as np
from keras.models import model_from_json
from experiment.qlearn import Catch

# time, to, well... keep track of time TODO: use it somewhere!
# Python image library for rendering
# IPython display for making sure we can render the frames
# seaborn for rendering
import seaborn

# Setup matplotlib so that it runs nicely in iPython
# %matplotlib inline
# setting up seaborn
seaborn.set()

if __name__ == "__main__":
    # Make sure this grid size matches the value used for training
    grid_size = 10

    with open("./mdl/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("./mdl/model.h5")
    model.compile("sgd", "mse")

    # Define environment, game
    env = Catch(grid_size)
    c = 0
    for e in range(10):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        plt.imshow(input_t.reshape((grid_size,) * 2),
                   interpolation='none', cmap='gray')
        plt.show()
        plt.savefig("./figs/%03d.png" % c)
        c += 1
        while not game_over:
            input_tm1 = input_t

            # get next action
            q = model.predict(input_tm1)
            action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            plt.imshow(input_t.reshape((grid_size,) * 2),
                       interpolation='none', cmap='gray')
            plt.show()
            plt.savefig("./figs/%03d.png" % c)
            c += 1
