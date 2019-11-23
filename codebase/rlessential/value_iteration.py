import os
import time

import numpy as np
import pandas as pd

from rlessential.mdp import TwoStateMDP
from util import timing

os.getcwd()
# os.chdir('../..')

discount = 0.95
epsilon = 0.01

threshold = (epsilon * (1 - discount)) / (2 * discount)


mdp = pd.read_csv('../dataset/mdp/twostate_mdp.csv')

two_state = TwoStateMDP(mdp)
num_s = two_state.num_states
num_a = two_state.num_actions


def value_iteration():

    v_curr = np.zeros((num_s, 1))
    v_next = np.full((num_s, 1), np.inf)

    r = np.full((num_s * num_a, 1), -np.inf)
    p = np.zeros((num_s * num_a, num_s))

    for index, row in mdp.iterrows():
        s = row['idstatefrom'].astype(int)
        a = row['idaction'].astype(int)
        sp = row['idstateto'].astype(int)

        r[s * num_a + a] = row['reward']
        p[s * num_a + a][sp] = row['probability']

    i = 0
    values = []
    t = time.perf_counter()
    while np.linalg.norm(v_next - v_curr, ord=np.inf) >= threshold:
        if i == 0:
            values.append((i, v_curr.reshape(num_s)))
            i += 1
        else:
            v_curr = v_next
        v = r + discount * p @ v_curr
        split_values = np.split(v, num_s)

        v_next = np.array(list(map(np.max, split_values)))
        v_next = v_next.reshape(num_s, 1)
        values.append((i, v_next.reshape(num_s), np.linalg.norm(v_next - v_curr, ord=np.inf)))
        i += 1

    timing(t)
    print(*values, sep='\n')
    print("threshold: ", threshold)
    timing(t)

    last_v = r + discount * p @ v_curr
    split_last_values = np.split(last_v, num_s)

    policy = np.array(list(map(np.argmax, split_last_values)))
    policy = policy.reshape(num_s, 1)
    print('the policy: \n', policy)
    timing(t)


def main():
    value_iteration()


if __name__ == '__main__':
    main()

