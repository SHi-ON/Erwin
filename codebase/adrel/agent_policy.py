import numpy as np
import pandas as pd

FILES_NAME = {'policy': '/policy_nn.csv',
              'scale': '/scales.csv',
              'q_values': '/qvalues.csv'}


class AgentPolicy:

    def __init__(self, policy_info):
        self.policy, self.scale, self.q_values = self.load_policy(policy_info['dir'],
                                                                  policy_info['q'])

        self.state_features = self.policy.values[:, 0:4]
        self.state_ids = np.array(self.policy['State'])
        self.actions = np.array(self.policy['Action'])
        self.values = np.array(self.policy['Value'])
        if 'Probability' in self.policy.columns:
            print('Stochastic policy loaded')
            self.probabilities = np.array(self.policy['Probability'])
        else:
            print('Deterministic policy loaded')
            self.probabilities = None

    @staticmethod
    def load_policy(directory, is_q):
        policy = pd.read_csv(directory + FILES_NAME['policy'])
        scale = pd.read_csv(directory + FILES_NAME['scale']).values
        q_values = pd.read_csv(directory + FILES_NAME['q_values']) if is_q else None
        return policy, scale, q_values

    def select_state(self, state, pert_noise):
        # scale the observed state
        state_scaled = state @ self.scale
        # TODO: multiplication w/o eye matrix
        perturbed = np.eye(self.scale.shape[0]) * (1 + pert_noise)
        state_scaled = state_scaled @ perturbed
        # ||(Sc + eps) - Sc||
        distance = np.linalg.norm(
            self.state_features - np.repeat(np.atleast_2d(state_scaled), self.state_features.shape[0], 0),
            axis=1)
        # find the closest state to the observed state in the policy
        # state index in the file, not the state number
        return np.argmin(distance)

    def select_action(self, state_index):
        if self.probabilities is not None:
            # find all relevant state ids
            all_state_ids = np.where(self.state_ids == self.state_ids[state_index])[0]
            all_probs = self.probabilities[all_state_ids]
            all_actions = self.actions[all_state_ids]
            assert (abs(1 - sum(all_probs)) < 0.01)
            action = int(np.random.choice(all_actions, p=all_probs))
        else:
            # assume that there is a single action for each state
            # switch between q-values or the policy
            if self.q_values is not None:
                # TODO: np.where rather than .loc[]
                state_q = self.q_values.loc[self.q_values['idstate'] == state_index]
                action = int(state_q.loc[state_q['qvalue'] == state_q.max()[2]]['idaction'])
            else:
                action = int(self.actions[state_index])
        return action
