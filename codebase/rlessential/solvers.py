import os
import time

import numpy as np
import pandas as pd
import gym

from rlessential.domains import TwoStateMDP, TwoStateParametricMDP
from util import timing

os.getcwd()
# os.chdir('../..')


def value_iteration(mdp, threshold, discount, initial_values=None):
    num_s = mdp.num_states
    num_a = mdp.num_actions
    r = mdp.get_rewards()
    p = mdp.get_probabilities()

    print('Value iteration begins')

    # set starting state values
    if initial_values is not None:
        v_curr = initial_values.reshape(num_s, 1)
    else:
        v_curr = np.zeros((num_s, 1))

    distance = np.inf
    values = list()
    values.append((0, v_curr.reshape(num_s)))
    i = 1
    t = time.perf_counter()
    while distance >= threshold:
        v = r + discount * p @ v_curr

        split_values = np.split(v, num_s)
        # maximizing the Bellman eq. for all actions
        v_next = np.array(list(map(np.max, split_values)))
        v_next = v_next.reshape(num_s, 1)
        distance = np.linalg.norm(v_next - v_curr, ord=np.inf)
        values.append((i,
                       v_next.reshape(num_s),
                       distance))
        i += 1
        v_curr = v_next

    timing(t)
    print(*values, sep='\n')

    # policy calculation
    last_v = r + discount * p @ v_curr
    split_last_values = np.split(last_v, num_s)
    # find actions that maximize the Bellman eq. (argmax)
    policy = np.array(list(map(np.argmax, split_last_values)))
    policy = policy.reshape(num_s, 1)
    print('value iteration - policy: \n', policy)
    timing(t)

    return values


def policy_iteration(mdp, threshold, discount):
    num_s = mdp.num_states
    num_a = mdp.num_actions
    r = mdp.get_rewards()
    p = mdp.get_probabilities()

    print('Policy iteration begins')

    v_curr = np.zeros((num_s, 1))
    policy_curr = np.zeros((num_s, 1))
    policy_next = np.full((num_s, 1), np.inf)

    t = time.perf_counter()
    while not np.array_equal(policy_curr, policy_next):
        distance = np.inf
        values = list()
        values.append((0, v_curr.reshape(num_s)))
        i = 1
        while distance >= threshold:
            v = r + discount * p @ v_curr

            split_values = np.split(v, num_s)
            # maximizing the Bellman eq. for all actions
            v_next = np.array(list(map(np.max, split_values)))
            v_next = v_next.reshape(num_s, 1)
            distance = np.linalg.norm(v_next - v_curr, ord=np.inf)
            values.append((i,
                           v_next.reshape(num_s),
                           distance))
            i += 1
            v_curr = v_next

        print(*values, sep='\n')

        # policy improvement
        last_v = r + discount * p @ v_curr
        split_last_values = np.split(last_v, num_s)
        # find actions that maximize the Bellman eq. (argmax)
        policy_curr = policy_next
        policy_next = np.array(list(map(np.argmax, split_last_values)))
        policy_next = policy_next.reshape(num_s, 1)
        print('policy iteration - policy: \n', policy_next)
    timing(t)

    return policy_next


def main():
    gamma = 0.95
    epsilon = 0.01
    tau = (epsilon * (1 - gamma)) / (2 * gamma)

    # mdp_data = pd.read_csv('../dataset/mdp/twostate_mdp_6-4-2.csv')
    # mdp = TwoStateParametricMDP(mdp_data, 0)
    # init_v = np.array([-4.5, -5])
    mdp_data = pd.read_csv('../dataset/mdp/twostate_mdp_3-1-1.csv')
    mdp = TwoStateMDP(mdp_data)

    # value_iteration(mdp, threshold=tau, discount=gamma, initial_values=init_v)
    value_iteration(mdp, threshold=tau, discount=gamma)
    policy_iteration(mdp, threshold=tau, discount=gamma)


if __name__ == '__main__':
    main()