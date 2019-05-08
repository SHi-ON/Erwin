import numpy as np
import gym
import tqdm
import time


class CartPolePlayer:

    def __init__(self, policy, scale, q_values=None, render=False):
        self.policy = policy
        self.scale = scale
        self.q_values = q_values

        self.state_features = policy.values[:, 0:4]
        self.state_ids = np.array(policy["State"])
        self.actions = np.array(policy["Action"])
        self.values = np.array(policy["Value"])
        if "Probability" in policy.columns:
            print("Using a randomized policy")
            self.policy_randomized = True
            self.probabilities = np.array(policy["Probability"])
        else:
            print("Using a deterministic policy")
            self.policy_randomized = False
            self.probabilities = None

        self.render = render

    def select_action(self, state):
        state_q = self.q_values.loc[self.q_values['idstate'] == state]
        action = state_q.loc[state_q['qvalue'] == state_q.max()[2]]['idaction']
        return int(action)

    def play(self, trials, eps):

        # gym/envs/__init__.py'
        # different max_episode_steps 'CartPole-v0' and 'CartPole-v1'
        env = gym.make('CartPole-v1')
        max_episode_steps = 200

        total_rewards = dict()

        for trial in tqdm.trange(trials):
            env.reset()
            done = False

            # take an initial random action and observe the state
            action = env.action_space.sample()

            [state, reward, done, info] = env.step(action)
            if done:
                continue

            for i in range(1, max_episode_steps):
                # faster with no graphical output
                if self.render:
                    env.render()
                    time.sleep(0.05)

                # scale the observed state
                state_scaled = state @ self.scale
                # TODO: multiplication w/o eye matrix
                perturbation = np.eye(self.scale.shape[0]) * (1 + eps)
                state_scaled = state_scaled @ perturbation
                # ||(Sc + eps) - Sc||
                distance = np.linalg.norm(
                    self.state_features - np.repeat(np.atleast_2d(state_scaled), self.state_features.shape[0], 0),
                    axis=1)
                # find the closest state to the observed state in the policy
                # state index in the file, not the state number
                state_index = np.argmin(distance)

                if self.policy_randomized:
                    id_state = self.state_ids[state_index]
                    # find all relevant state ids
                    all_state_ids = np.where(self.state_ids == id_state)[0]
                    all_probs = self.probabilities[all_state_ids]
                    all_actions = self.actions[all_state_ids]
                    assert (abs(1 - sum(all_probs)) < 0.01)
                    action = int(np.random.choice(all_actions, p=all_probs))
                else:
                    # assume that there is a single action for each state
                    # switch between q-learning or the policy
                    if self.q_values is not None:
                        action = self.select_action(state_index)
                    else:
                        action = int(self.actions[state_index])

                # stop only after saving the state
                if done:
                    break

                [state, reward, done, info] = env.step(action)  # take a random action

                if trial in total_rewards:
                    total_rewards[trial] += reward
                else:
                    total_rewards[trial] = reward

        env.close()
        return total_rewards
