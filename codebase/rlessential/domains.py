from abc import ABC, abstractmethod

import numpy as np

from sample import Sample

# TODO: migrate self-defined domains in lspi's domain module to here

COL_STATE_FROM = 'idstatefrom'
COL_ACTION = 'idaction'
COL_STATE_TO = 'idstateto'
COL_PROBABILITY = 'probability'
COL_REWARD = 'reward'


class MDP(ABC):
    # new in Python 3.4:
    # __metaclass__ = ABCMeta

    @abstractmethod
    def state_count(self):
        """
        Counts the states in the MDP

        :return: number of states
        """
        pass  # pragma: no cover

    @abstractmethod
    def action_count(self):
        """
        Counts the actions in the MDP

        :return: number of actions
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_rewards(self):
        """
        Calculates the reward vector.
        dimensionality: :math:`|S||A|`

        :return: the reward matrix
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_probabilities(self):
        """
        Calculates the transition probabilities matrix.
        dimensionality: :math:`|S||A| * |S|`

        :return: the transition probabilities matrix
        """
        pass  # pragma: no cover

    @abstractmethod
    def step(self, action):
        """
        Applies the action and returns the observed sample.

        :arg action Action index. Must be in range [0, num_actions)

        :return: observed sample
        """

        pass  # pragma: no cover

    @abstractmethod
    def reset(self, initial_state):
        """
        Resets the domain to the initial state to start from.

        :arg initial_state the starting state
        """

        pass  # pragma: no cover


class BaseMDP(MDP):

    def __init__(self, mdp, initial_state=None):
        self.mdp = mdp
        self.num_states = self.state_count()
        self.num_actions = self.action_count()
        self.probs = self.get_probabilities()
        self.rewards = self.get_rewards()

        self.state_ = None
        self.reset(initial_state)

    def state_count(self):
        unique_from = self.mdp[COL_STATE_FROM].nunique()
        unique_to = self.mdp[COL_STATE_TO].nunique()
        return max(unique_from, unique_to)

    def action_count(self):
        unique_action = self.mdp[COL_ACTION].nunique()
        return unique_action

    def get_probabilities(self):
        probs = np.zeros((self.num_states * self.num_actions, self.num_states))
        for index, row in self.mdp.iterrows():
            s = row[COL_STATE_FROM].astype(np.int)
            a = row[COL_ACTION].astype(np.int)
            sp = row[COL_STATE_TO].astype(np.int)
            # any float value, no type casting
            probs[s * self.num_actions + a][sp] = row[COL_PROBABILITY]
        return probs

    def get_rewards(self):
        rewards = np.zeros((self.num_states * self.num_actions, 1))
        for index, row in self.mdp.iterrows():
            s = row[COL_STATE_FROM].astype(np.int)
            a = row[COL_ACTION].astype(np.int)
            # any float value, no type casting
            rewards[s * self.num_actions + a] = row[COL_REWARD]
        return rewards

    def get_allowed_actions(self, state=None):
        if state is None:
            state = self.state_

        candidate_transitions = self.mdp.loc[self.mdp[COL_STATE_FROM] == state.item()]
        candidate_transitions = candidate_transitions.loc[candidate_transitions[COL_PROBABILITY] != 0]
        if len(candidate_transitions) == 0:
            raise KeyError('Not able to find any candidates for state {}'.format(state))

        allowed_actions = candidate_transitions[COL_ACTION].unique()
        return allowed_actions

    def step(self, action):
        if action < 0 or action > self.num_actions - 1:
            raise IndexError('Action index outside of bound [0, %d)'.format(self.num_actions))

        candidate_transitions_condition = (self.mdp[COL_STATE_FROM] == self.state_.item()) \
                                          & (self.mdp[COL_ACTION] == action)
        candidate_transitions = self.mdp.loc[candidate_transitions_condition]
        if len(candidate_transitions) == 0:
            raise KeyError('Not able to find any candidates for state-action pair ({}, {})'.format(self.state_, action))
        elif len(candidate_transitions) != 1:
            candidate_transitions = candidate_transitions.sample(weights=COL_PROBABILITY)

        transition_reward = candidate_transitions[COL_REWARD].values.reshape(1, )
        transition_next_state = candidate_transitions[COL_STATE_TO].values.reshape(1, ).astype(np.int)

        sample = Sample(self.state_, action, transition_reward, transition_next_state)

        self.state_ = transition_next_state

        return sample

    def reset(self, initial_state=None):
        if initial_state is None:
            self.state_ = np.array([0])
        elif initial_state < 0 or initial_state > self.num_states - 1:
            raise IndexError('State index outside of bound [0, %d)'.format(self.num_states))
        elif initial_state.shape != (1,):
            raise ValueError('Initial state shape mismatch')
        else:
            try:
                state = initial_state.astype(np.int)
                self.state_ = state
            except ValueError:
                print('Not a valid integer')


class TwoStateMDP(BaseMDP):
    """
    MDP from Figure 3.1.1 in Putterman's MDP book, page 34.

    **Details**:

    2 states: {s1, s2} -> {0, 1}

    3 actions: {a11, a12, a21} -> {0, 1, 2}
    """

    def __init__(self, mdp):
        # noinspection PyCompatibility
        super().__init__(mdp)


class TwoStateParametricMDP(BaseMDP):
    """
    MDP from example 6.4.2 in Putterman's MDP book, page 182.

    **Details**:

    2 states: {s1, s2} -> {0, 1}

    2 actions: {a, a2,1} -> {0, 1}

    2 rewards: {:math:`-a^2`, -0.5} -> {0, 1}

    3 probabilities: {a/2, 1-a/2, 1} -> {0, 1, 2}
    """

    def __init__(self, mdp, param):
        # noinspection PyCompatibility
        super().__init__(mdp)
        self.param = param
        self.rewards = TwoStateParametricMDP.parametrize_rewards(param)
        self.probabilities = TwoStateParametricMDP.parametrize_probabilities(param)

    def get_rewards(self):
        rewards = np.full((self.num_states * self.num_actions, 1), -np.inf)
        for index, row in self.mdp.iterrows():
            s = row[COL_STATE_FROM].astype(int)
            a = row[COL_ACTION].astype(int)
            # any float value, no type casting
            reward_index = row[COL_REWARD]
            rewards[s * self.num_actions + a] = self.rewards[reward_index]
        return rewards

    def get_probabilities(self):
        probs = np.zeros((self.num_states * self.num_actions, self.num_states))
        for index, row in self.mdp.iterrows():
            s = row[COL_STATE_FROM].astype(int)
            a = row[COL_ACTION].astype(int)
            sp = row[COL_STATE_TO].astype(int)
            # any float value, no type casting
            prob_index = row[COL_PROBABILITY]
            probs[s * self.num_actions + a][sp] = self.probabilities[prob_index]
        return probs

    @staticmethod
    def parametrize_rewards(a):
        return [- a ** 2, -0.5]

    @staticmethod
    def parametrize_probabilities(a):
        return [a / 2, 1 - a / 2, 1]


class RAAMMDP(BaseMDP):
    """
    Three-state deterministic MDP in the `RAAM Paper <http://www.cs.unh.edu/~mpetrik/pub/Petrik2014_appendix.pdf>`_.

    **Details**:

    3 states: {s1, s2, s3} -> {0, 1, 2}

    3 actions: {a1, a2, 0} -> {0, 1, 2}

    3 rewards: {0, 1, :math:`\epsilon`} -> {0, 1, 2}

    3 probabilities: deterministic (all ones)

    """

    def __init__(self, mdp):
        # noinspection PyCompatibility
        super().__init__(mdp)


class MachineReplacementMDP(BaseMDP):
    """
    Machine Replacement MDP problem from the `Percentile Optimization paper
    <http://web.hec.ca/pages/erick.delage/percentileMDP.pdf>`_, Figure 3.

    **Details**:

    10 states: {1, 2, ..., 8, R1, R2} -> {0, 1, 2, ..., 9}

    5 actions: 3 **repairs** and 2 **do nothings**

    """

    def __init__(self, mdp):
        # noinspection PyCompatibility
        super().__init__(mdp)
