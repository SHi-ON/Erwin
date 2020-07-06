import time

import numpy as np

from util import timing


class ValueIterationSolver:

    def __init__(self, domain, discount, threshold,
                 max_iterations=0, initial_values=None, verbose=False):
        self.domain = domain

        self.discount = discount
        self.threshold = threshold

        self.initial_values = initial_values
        self.max_iterations = max_iterations
        self.verbose = verbose

        # format: (#iter, V(s), distance)
        self.iter_values = list()
        self.policy = None

    def get_value_table(self):
        if not self.iter_values:
            print('Value iteration has not been run yet!')
            self.calculate_values()

        num_s = self.domain.num_states
        last_v = self.iter_values[-1][1].reshape(num_s, 1)
        return last_v

    def calculate_values(self):
        num_s = self.domain.num_states
        num_a = self.domain.num_actions
        r = self.domain.get_rewards()
        p = self.domain.get_probabilities()
        sum_probs = np.sum(p, axis=1, keepdims=True)

        if self.verbose:
            print('Value iteration begins...')

        # set starting state values
        if self.initial_values is not None:
            v_curr = self.initial_values.reshape(num_s, 1)
        else:
            v_curr = np.zeros((num_s, 1))

        self.iter_values.append((0, v_curr.reshape(num_s)))

        dist = np.inf
        i = 1
        t = time.perf_counter()
        while dist >= self.threshold or i < self.max_iterations:

            # noinspection PyCompatibility
            v = r + self.discount * p @ v_curr

            split_values = np.split(v, num_s)
            split_values = np.array(split_values)
            for s in range(num_s):
                for a in range(num_a):
                    if sum_probs[s * num_a + a] == 0:
                        split_values[s][a] = -np.inf
            # maximizing the Bellman eq. for all actions
            v_next = np.array(list(map(np.max, split_values)))
            v_next = v_next.reshape(num_s, 1)
            dist = np.linalg.norm(v_next - v_curr, ord=np.inf)
            self.iter_values.append((i,
                                     v_next.reshape(num_s),
                                     dist))
            i += 1
            v_curr = v_next

        if self.verbose:
            timing(t)
            print('Value iteration finished:')
            print(*self.iter_values, sep='\n')

        values = self.get_value_table()
        return values

    def calculate_policy(self):
        if not self.iter_values:
            print('Value iteration has not been run yet!')
            self.calculate_values()
        num_s = self.domain.num_states
        num_a = self.domain.num_actions
        r = self.domain.get_rewards()
        p = self.domain.get_probabilities()
        sum_probs = np.sum(p, axis=1, keepdims=True)

        last_v = self.get_value_table()
        # noinspection PyCompatibility
        last_v = r + self.discount * p @ last_v
        split_last_values = np.split(last_v, num_s)
        split_last_values = np.array(split_last_values)
        for s in range(num_s):
            for a in range(num_a):
                if sum_probs[s * num_a + a] == 0:
                    split_last_values[s][a] = -np.inf

        policy_calc = np.array(list(map(np.argmax, split_last_values)))
        policy_calc = policy_calc.reshape(num_s, 1)
        if self.verbose:
            print('value iteration - calculated policy: \n', policy_calc)

        self.policy = policy_calc
        return policy_calc


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
