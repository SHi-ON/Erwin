import abc

import numpy as np
import gym
import tqdm

from sample import Sample


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

    def run(self, render=True):
        print('\nAgent run started ...')
        total_reward = 0
        for e in tqdm.trange(self.num_episodes):
            current_state = self.env.reset()
            current_state = self.aggregate(current_state)

            done = False
            while not done:
                if render:
                    self.env.render()
                action = np.argmax(self.q_table[current_state])
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = self.aggregate(observation)
                current_state = next_state

        avg_reward = total_reward / self.num_episodes
        print('Average reward: {0} in {1} episodes'.format(avg_reward, self.num_episodes))
        print('Agent run finished!')

    def simulate(self):
        print('\nSimulation started ...')

        for e in tqdm.trange(self.num_episodes):
            current_observation = self.env.reset()
            current_state = self.aggregate(current_observation)

            done = False
            while not done:
                sample = list()
                sample.append(current_observation)
                action = self.choose_action(current_state)
                sample.append(action)
                next_observation, reward, done, _ = self.env.step(action)
                sample.append(reward)
                sample.append(next_observation)
                next_state = self.aggregate(next_observation)
                current_observation = next_observation
                current_state = next_state
                self.batch.append(tuple(sample))
        print('Simulation finished!')


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
        self.batch = list()


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

    def __init__(self, samples, env, num_buckets=(1, 40, 40, 40)):
        self.samples = samples
        self.env = env
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
        self.A = list()
        self.B = list()

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
        for sample in self.samples:
            # sample: (s, a, r, s')
            current_state = self.aggregate(sample[0])
            action = sample[1]
            next_state = self.aggregate(sample[3])

            # FIXME: consider as a set of --unique-- actions and outcomes
            if current_state not in self.S:
                self.S.append(current_state)
            if next_state not in self.S:
                self.S.append(next_state)
            if action not in self.A:
                self.A.append(action)
            if next_state not in self.B:
                self.B.append(next_state)

    def compute(self):
        sample_size = len(self.S)
        action_size = len(self.A)
        outcome_size = len(self.B)
        self.P = np.zeros((action_size, outcome_size, sample_size, sample_size))
        self.r = np.zeros((action_size, outcome_size, sample_size))
        for s in self.S:
            for s_p in self.S:
                for a in self.A:
                    for b in self.B:
                        transition_reward = list()
                        for sample in self.samples:
                            if sample[0] == b and sample[1] == a:
                                transition_reward.append((sample[3], sample[2]))
                        sum_states = 0
                        for s_r in transition_reward:
                            if s_p == s_r[0]:
                                sum_states += 1
                        self.P[a, b, s, s_p] = (1 / len(transition_reward)) * sum_states
                        sum_rewards = 0
                        for s_r in transition_reward:
                            sum_rewards += s_r[1]
                        # TODO: check to be correct
                        self.r[a, b, s] = (1 / len(transition_reward)) * sum_rewards


class MachineReplacementMDPAgent(Agent):

    def __init__(self, domain, solver, discount, horizon=None):
        self.domain = domain
        self.solver = solver
        self.discount = discount

        self.horizon = horizon
        self.state_ = 0

    def choose_action(self, state):
        # TODO: try epsilon-greedy or Boltzmann distribution

        allowed_actions = self.domain.get_allowed_actions()

        # randomized action selection - uniform distribution
        return np.random.choice(allowed_actions)

    def train(self):
        self.solver.calculate_value()

    def run(self):
        curr_state = self.domain.state_
        samples = list()
        i = 0
        while i < self.horizon:
            action = self.choose_action(curr_state)
            sample = self.domain.step(action)
            curr_state = sample.next_state
            samples.append(sample)
            i += 1


# showcase the agent performance
if __name__ == '__main__':
    agent = QLearningCartPoleAgent()
    agent.train()
    agent.run()
