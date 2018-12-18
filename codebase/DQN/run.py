import numpy as np
import pandas as pd
from keras.models import model_from_json
import matplotlib.pyplot as plt

from qtrain import *
from util import *
import sample as samp


def model_import():
    with open("./models/" + model_name + ".json", "r") as m_file:
        mdl = model_from_json(json.load(m_file))
        mdl.load_weights("./models/" + model_name + ".h5")
        mdl.compile("adam", "mse")  # TODO
    return mdl


def policy_store(df, plc):
    plc_df = pd.DataFrame(plc)
    new_df = pd.concat([df, plc_df], axis=1, sort=False)
    new_df.columns = policy_col_names
    new_df.to_csv("./datasets/" + policy_name + samples_name + ".csv", sep=",")


def run_model():
    d_f = handler()
    smpl = samp.Sample(d_f, v1_max, reward_col)
    epochs = smpl.get_epochs()

    policy = np.zeros((epochs * v1_max, 1))

    model = model_import()

    for ep in range(epochs):
        loss = 0.

        next_state = smpl.get_init_sample(ep)
        game_over = smpl.is_over(ep)

        # plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
        # plt.imshow(next_state, interpolation='none', cmap='gray')
        # plt.show()
        # plt.savefig("./figs/%03d.png" % ep)

        for i in range(1, v1_max):
            state = next_state

            q = model.predict(state)
            # action = np.argmax(q[0])
            action = np.argmax(q[0])
            # if action == 1:
            #     print("Action in ", ep, "and step", i)
            policy[ep * v1_max + i] = pick_action(q[0])

            next_state = smpl.get_sample(ep, i)
            game_over = smpl.is_over(ep)

            # plt.imshow(input_t.reshape((grid_size,) * 2), interpolation='none', cmap='gray')
            # plt.imshow(next_state, interpolation='none', cmap='gray')
            # plt.show()
            # plt.savefig("./figs/%03d.png" % (ep + 1))

    policy_store(d_f, policy)


def moving_average_diff(a, n=100):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    run_model()
    print("Finished")


    # plt.plot(moving_average_diff(hist))
    # plt.ylabel('Average of victories per game')
    # plt.show()
