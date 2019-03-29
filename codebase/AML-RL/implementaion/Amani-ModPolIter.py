import sys
import pandas as pd
import numpy as np
from implementaion.polimod import *

if __name__ == "__main__":
    input_values = pd.read_csv(sys.argv[1], header=None,
                               names=['idstatefrom', 'idaction', 'idstateto', 'probability', 'reward'])
    # input_values = np.genfromtxt(sys.argv[1], delimiter=',')
    # print(input_values)


    # print(input_values.shape)
    # print(input_values[['probability']].shape)
    trans_probs = np.array([[[0.3, 0.2, 0.1, 0.4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
    rewards = np.array([[[1.0, 1.5, 1.8, 2.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    # mvi = PolicyIterationModified(trans_probs, rewards, 0.95)
    # mdp.PolicyIteration(input_values[],input_values[['reward']],0.95)
    #
    # mvi.run()
    # print(mvi.policy)
    # print(mvi.V)

    arr = np.array([2, 3, 5, 2])


