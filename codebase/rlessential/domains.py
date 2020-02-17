import abc

import numpy as np


class MDP(object):
    __metaclass__ = abc.ABCMeta

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


class TwoStateMDP(MDP):
    """
    MDP from Figure 3.1.1 in Putterman's MDP book, page 34.

    **Details**:

    2 states: {s1, s2} -> {0, 1}

    3 actions: {a11, a12, a21} -> {0, 1, 2}
    """

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
            # any float value, no type casting
            rewards[s * self.num_actions + a] = row['reward']
        return rewards

    def get_probabilities(self):
        probs = np.zeros((self.num_states * self.num_actions, self.num_states))
        for index, row in self.mdp.iterrows():
            s = row['idstatefrom'].astype(int)
            a = row['idaction'].astype(int)
            sp = row['idstateto'].astype(int)
            # any float value, no type casting
            probs[s * self.num_actions + a][sp] = row['probability']
        return probs


class TwoStateParametricMDP(MDP):
    """
    MDP from example 6.4.2 in Putterman's MDP book, page 182.

    **Details**:

    2 states: {s1, s2} -> {0, 1}

    2 actions: {a, a2,1} -> {0, 1}

    2 rewards: {:math:`-a^2`, -0.5} -> {0, 1}

    3 probabilities: {a/2, 1-a/2, 1} -> {0, 1, 2}
    """

    def __init__(self, mdp, param):
        self.mdp = mdp
        self.param = param
        self.num_states = self.state_count()
        self.num_actions = self.action_count()
        self.rewards = TwoStateParametricMDP.parametrize_rewards(param)
        self.probabilities = TwoStateParametricMDP.parametrize_probabilities(param)

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
            # any float value, no type casting
            reward_index = row['reward']
            rewards[s * self.num_actions + a] = self.rewards[reward_index]
        return rewards

    def get_probabilities(self):
        probs = np.zeros((self.num_states * self.num_actions, self.num_states))
        for index, row in self.mdp.iterrows():
            s = row['idstatefrom'].astype(int)
            a = row['idaction'].astype(int)
            sp = row['idstateto'].astype(int)
            # any float value, no type casting
            prob_index = row['probability']
            probs[s * self.num_actions + a][sp] = self.probabilities[prob_index]
        return probs

    @staticmethod
    def parametrize_rewards(a):
        return [- a ** 2, -0.5]

    @staticmethod
    def parametrize_probabilities(a):
        return [a / 2, 1 - a / 2, 1]


class RAAMMDP(MDP):
    """
    Three-state deterministic MDP in the `RAAM Paper <http://www.cs.unh.edu/~mpetrik/pub/Petrik2014_appendix.pdf>`_.

    **Details**:

    3 states: {s1, s2, s3} -> {0, 1, 2}

    3 actions: {a1, a2, 0} -> {0, 1, 2}

    3 rewards: {0, 1, :math:`\epsilon`} -> {0, 1, 2}

    3 probabilities: deterministic (all ones)

    """

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


class MachineReplacementMDP(MDP):

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
        rewards = np.zeros((self.num_states * self.num_actions, 1))
        for  index, row in self.mdp.iterrows():
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
