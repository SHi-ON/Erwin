import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import *
import sample as samp


def run_model():
    d_f = dataset_import()
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

    policy_export(d_f, policy)


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
