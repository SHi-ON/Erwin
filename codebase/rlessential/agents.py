import abc

import numpy as np
import gym


class Agent(object):
    """
    Agents prototype

    The agent has the environment embedded in itself for consistency reasons.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def choose_action(self, state):
        """
        Action selection by following exploration-exploitation trade-off.

        :return: selected action from action space.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def train(self):
        pass  # pragma: no cover

    @abc.abstractmethod
    def run(self):
        pass  # pragma: no cover


class QLearningCartPoleAgent(Agent):
    """
    Q-learning in OpenAI Gym CartPole environment.
    The default is an undiscounted problem.
    Initialized the agent with the desired discount parameter.
    """

    def __init__(self,
                 num_buckets=(1, 1, 6, 12),
                 num_episodes=1000,
                 discount=1.0,
                 min_explore=0.1,
                 min_learning=0.1,
                 decay=25):
        self.num_buckets = num_buckets
        self.num_episodes = num_episodes
        self.discount = discount
        self.min_explore = min_explore
        self.min_learning = min_learning
        self.decay = decay

        self.env = gym.make('CartPole-v1')

        self.explore = self.get_explore(0)
        self.learning = self.get_learning(0)

        self.upper_bounds = [self.env.observation_space.high[0],
                             0.5,
                             self.env.observation_space.high[2],
                             np.radians(50) / 1]
        self.lower_bounds = [self.env.observation_space.low[0],
                             -0.5,
                             self.env.observation_space.low[2],
                             -np.radians(50) / 1]

        self.q_table = np.zeros(self.num_buckets + (self.env.action_space.n,))

    def aggregate_state(self, obs):
        aggregate = list()
        for i in range(len(obs)):
            scale = (obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            scaled_obs = int(np.round((self.num_buckets[i] - 1) * scale))
            # fit the scaled observation within the first and last bucket
            scaled_obs = min(self.num_buckets[i] - 1,
                             max(0, scaled_obs))
            aggregate.append(scaled_obs)
        return tuple(aggregate)

    def choose_action(self, state):
        return self.env.action_space.sample() \
            if np.random.random() < self.explore \
            else np.argmax(self.q_table[state])

    def get_learning(self, t):
        return max(self.min_learning, min(1, 1 - np.log10((t + 1) / self.decay)))

    def get_explore(self, t):
        return max(self.min_explore, min(1, 1 - np.log10((t + 1) / self.decay)))

    def update_q(self, state, action, reward, next_state):
        self.q_table[state + (action,)] += self.learning * (reward +
                                                            self.discount * np.max(self.q_table[next_state]) -
                                                            self.q_table[state + (action,)])

    def train(self):
        print('\nTraining started ...')

        for e in range(self.num_episodes):
            current_state = self.env.reset()
            current_state = self.aggregate_state(current_state)

            self.explore = self.get_explore(e)
            self.learning = self.get_learning(e)

            done = False

            while not done:
                action = self.choose_action(current_state)
                observation, reward, done, _ = self.env.step(action)
                next_state = self.aggregate_state(observation)
                self.update_q(current_state, action, reward, next_state)
                current_state = next_state

        print('Training finished!')

    def run(self):
        while True:
            current_state = self.env.reset()
            current_state = self.aggregate_state(current_state)

            done = False
            while not done:
                self.env.render()
                action = np.argmax(self.q_table[current_state])
                observation, reward, done, _ = self.env.step(action)
                next_state = self.aggregate_state(observation)
                current_state = next_state


if __name__ == '__main__':
    agent = QLearningCartPoleAgent()
    agent.train()
    agent.run()
