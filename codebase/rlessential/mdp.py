import abc

import numpy as np


class MDP(object):

    @abc.abstractmethod
    def state_count(self):
        """
        Counts the states in the MDP

        :return: number of states
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def action_count(self):
        """
        Counts the actions in the MDP

        :return: number of actions
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def get_rewards(self):
        """
        Calculates the reward vector.
        dimensionality: :math:`|S||A|`

        :return: the reward matrix
        """

        pass  # pragma: no cover

    @abc.abstractmethod
    def get_probabilities(self):
        """
        Calculates the transition probabilities matrix.
        dimensionality: :math:`|S||A| * |S|`

        :return: the transition probabilities matrix
        """
        pass  # pragma: no cover


class TwoStateMDP:

    def __init__(self, mdp):
        self.mdp = mdp
        self.num_states = self.state_count()
        self.num_actions = self.action_count()

    def state_count(self):
        unique_from = self.mdp['idstatefrom'].nunique()
        unique_to = self.mdp['idstateto'].nunique()
        return max(unique_from, unique_to)

    def action_count(self):
        unique_action = self.mdp['idaction'].nunique()
        return unique_action

    def get_rewards(self):
        rewards = np.full((self.num_states * self.num_actions, 1), -np.inf)
        for index, row in self.mdp.iterrows():
            s = row['idstatefrom'].astype(int)
            a = row['idaction'].astype(int)
            rewards[s * self.num_actions + a] = row['reward']
        return rewards

    def get_probabilities(self):
        probs = np.zeros((self.num_states * self.num_actions, self.num_states))
        for index, row in self.mdp.iterrows():
            s = row['idstatefrom'].astype(int)
            a = row['idaction'].astype(int)
            sp = row['idstateto'].astype(int)
            probs[s * self.num_actions + a][sp] = row['probability']
        return probs


class TwoStateParametricMDP:
    __num_states = 2
    __num_actions = 2

    def __init__(self, param):
        self.num_states = TwoStateParametricMDP.__num_states
        self.num_actions = TwoStateParametricMDP.__num_actions
        self.param = param

    def state_count(self):
        return self.num_states

    def action_count(self):
        return self.num_actions

    def get_rewards(self):
        rewards = np.array([-self.param ** 2, -np.inf, -np.inf, -0.5])
        rewards = rewards.reshape(self.num_states * self.num_actions, 1)
        return rewards

    def get_probabilities(self):
        probs = np.array([[self.param / 2, 1 - (self.param / 2)],
                          [0, 0],
                          [0, 0],
                          [0, 1]])
        probs = probs.reshape(self.num_states * self.num_actions, self.num_states)
        return probs
