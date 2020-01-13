import abc

import numpy as np
import gym
import tqdm


class Agent(object):
    """
    Agents prototype.

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


class QLearningAgent(Agent):
    """
    Q-Learning agent prototype with state aggregation based on OpenAI Gym environments.

    """

    def aggregate(self, obs):
        aggregate = list()
        for i in range(len(obs)):
            scale = (obs[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            scaled_obs = int(np.round((self.num_buckets[i] - 1) * scale))
            # exception management: fit the scaled observation within the first and last bucket
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

        for e in tqdm.trange(self.num_episodes):
            observation = self.env.reset()
            current_state = self.aggregate(observation)

            self.explore = self.get_explore(e)
            self.learning = self.get_learning(e)

            done = False

            while not done:
                action = self.choose_action(current_state)
                observation, reward, done, _ = self.env.step(action)
                next_state = self.aggregate(observation)
                self.update_q(current_state, action, reward, next_state)
                current_state = next_state

        print('Training finished!')

    def run(self):
        while True:
            current_state = self.env.reset()
            current_state = self.aggregate(current_state)

            done = False
            while not done:
                self.env.render()
                action = np.argmax(self.q_table[current_state])
                observation, reward, done, _ = self.env.step(action)
                next_state = self.aggregate(observation)
                current_state = next_state


class CartPoleAgent(Agent):
    """
    CartPole agent solely for sample collection.

    Agent samples trajectories on by taking random actions.

    """

    def __init__(self, num_episodes=1000):
        self.env = gym.make('CartPole-v1')
        self.num_episodes = num_episodes
        self.samples = list()

    def choose_action(self, state):
        pass

    def train(self):
        pass

    def run(self):
        """
        Collects samples in a tuple of (s, a, r, s').

        :return:
        """
        for e in tqdm.trange(self.num_episodes):
            observation = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_observation, reward, done, _ = self.env.step(action)
                sample = (observation, action, reward, next_observation)
                self.samples.append(sample)
                observation = next_observation


class QLearningCartPoleAgent(QLearningAgent):
    """
        Q-learning in OpenAI Gym CartPole environment.
        The default is an undiscounted problem.
        Initialize the agent with the desired discount parameter.
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


class QLearningMountainCarAgent(QLearningAgent):
    """
        Q-learning in OpenAI Gym MountainCar environment.
        The default is an undiscounted problem.
        Initialize the agent with the desired discount parameter.
    """

    def __init__(self,
                 num_buckets=(6, 12),
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

        self.env = gym.make('MountainCar-v0')

        self.explore = self.get_explore(0)
        self.learning = self.get_learning(0)

        self.upper_bounds = [self.env.observation_space.high[0],
                             self.env.observation_space.high[1]]

        self.lower_bounds = [self.env.observation_space.low[0],
                             self.env.observation_space.low[1]]

        self.q_table = np.zeros(self.num_buckets + (self.env.action_space.n,))


class RAAMAgent:

    def __init__(self, samples, num_buckets=(1, 40, 40, 40)):
        self.samples = samples
        self.num_buckets = num_buckets

        self.upper_bounds = [self.env.observation_space.high[0],
                             0.5,
                             self.env.observation_space.high[2],
                             np.radians(50) / 1]
        self.lower_bounds = [self.env.observation_space.low[0],
                             -0.5,
                             self.env.observation_space.low[2],
                             -np.radians(50) / 1]

        self.S = list()

    def aggregate(self, obs):
        aggregate = list()
        for i in range(len(obs)):
            scale = (obs[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i])
            scaled_obs = int(np.round((self.num_buckets[i] - 1) * scale))
            # exception management: fit the scaled observation within the first and last bucket
            scaled_obs = min(self.num_buckets[i] - 1,
                             max(0, scaled_obs))
            aggregate.append(scaled_obs)
        return tuple(aggregate)

    def aggregate_states(self):
        states = list()
        for sample in self.samples:
            current_state = self.aggregate(sample[0])
            next_state = self.aggregate(sample[3])
            if current_state not in states:
                states.append(current_state)
            if next_state not in states:
                states.append(next_state)
        self.S = states




# showcase the agent performance
if __name__ == '__main__':
    agent = QLearningCartPoleAgent()
    agent.train()
    agent.run()
