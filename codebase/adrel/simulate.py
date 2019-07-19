import time
import pickle

import numpy as np
import gym
import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns

from agent_policy import AgentPolicy
from agent_dqn import AgentDQN
from util import timing, pickle_store

# gym/envs/__init__.py'
# different max_episode_steps 'CartPole-v0' and 'CartPole-v1'
GYM_ENV = 'CartPole-v1'

SERIES_RUN = False
IS_RENDER = False
SAVE_LOAD = False

EPISODES = 10
SERIES_STEPS = 1
EPISODE_STEPS = 200
BATCH_SIZE = 16
PENALTY = -10

MODEL_NAME = './model/cartpole-dqn.h5'
POLICIES = [{'dir': 'initial', 'q': False},
            {'dir': 'qvals', 'q': True}]

""" CartPole-v0:

     Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
    Solved Requirements:
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive EPISODES.
"""


# CartPole parameters
#     self.masscart = 1.0 **
#     self.masspole = 0.1
#     self.length = 0.5  # actually half the pole's length **
#
#     self.polemass_length = (self.masspole * self.length)
#     self.total_mass = (self.masspole + self.masscart)
#
#     self.gravity = 9.8
#     self.force_mag = 10.0
#     self.tau = 0.02  # seconds between state updates
#     self.kinematics_integrator = 'euler'


def play_policy(policy_info, pert_noise):
    env = gym.make(GYM_ENV)
    agent = AgentPolicy(policy_info)

    game_reward = dict()
    for episode in tqdm.trange(EPISODES):
        env.reset()
        # take an initial random action and observe the state
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            continue
        for step in range(1, EPISODE_STEPS):
            # faster with no graphical output
            if IS_RENDER:
                env.render()
                # time.sleep(0.05)
            state_index = agent.select_state(state, pert_noise)
            action = agent.select_action(state_index)
            if done:
                game_reward[episode] = step + 1
                break
            state, reward, done, _ = env.step(action)
            if step == EPISODE_STEPS - 1:
                game_reward[episode] = step + 1
    env.close()
    return game_reward


def play_dqn(num_episodes):
    env = gym.make(GYM_ENV)
    space_size_state = env.observation_space.shape[0]
    space_size_action = env.action_space.n
    agent = AgentDQN(space_size_state, space_size_action)
    if SAVE_LOAD:
        agent.load(MODEL_NAME)
    _game_rewards = dict()
    for episode in range(1, num_episodes + 1):
        # returns the initial observation (state)
        state = env.reset()
        state = np.reshape(state, [1, space_size_state])
        for step in range(1, EPISODE_STEPS + 1):
            if IS_RENDER:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else agent.penalty
            next_state = np.reshape(next_state, [1, space_size_state])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {0}/{1}, reward: {2}, e: {3:.2}"
                      .format(episode, num_episodes, step, agent.epsilon))
                _game_rewards[episode] = step
                break
            # Experience Replay
            agent.replay(BATCH_SIZE)
            if step == EPISODE_STEPS:
                print("episode: {0}/{1}, reward: {2}, e: {3:.2}"
                      .format(episode, num_episodes, step, agent.epsilon))
                _game_rewards[episode] = step
        if SAVE_LOAD and episode % 10 == 0:
            agent.save(MODEL_NAME)
    env.close()
    return _game_rewards


if __name__ == "__main__":
    rewards_total = dict()

    print('*** DQN training...')
    t_start = time.perf_counter()
    for series_ep in range(1 if SERIES_RUN else EPISODES, EPISODES + 1, SERIES_STEPS):
        game_rewards = play_dqn(series_ep)
        game_reward_total = sum(game_rewards.values())
        game_reward_avg = game_reward_total / series_ep
        print("\tAverage reward per episode {:.2f}".format(game_reward_avg))
        rewards_total[series_ep] = game_reward_total
    t_finish = timing(t_start)

    sum_rewards_total = sum(rewards_total.values())
    time_per_step = t_finish / sum_rewards_total
    print('*** Time per step: {:.4f} second(s)'.format(time_per_step))
    print('*** Sum of total rewards: {}'.format(sum_rewards_total))

    pickle_store(rewards_total, 'rewards-total_1000-episodes')

    env = gym.make(GYM_ENV)

    #     self.gravity = 9.8
    #     self.masscart = 1.0
    #     self.masspole = 0.1
    #     self.total_mass = (self.masspole + self.masscart)
    #     self.length = 0.5  # actually half the pole's length
    #     self.polemass_length = (self.masspole * self.length)
    #     self.force_mag = 10.0
    #     self.tau = 0.02  # seconds between state updates
    #     self.kinematics_integrator = 'euler'

    env.masscart = 1.0
    env.length = 0.5

    # len(rewards)
    #
    # average_reward
    # len(avg_rewards)
    #
    # len(x_axis)
    # len(y_axis)
    #
    # x_axis = np.array([i for i in range(0, EPISODES)])
    # y_axis = np.array(avg_rewards)
    # x_axis.shape
    # y_axis.shape
    #
    # x_axis = [value for index, value in enumerate(x_axis) if index % 5 == 0]
    # y_axis = [value for index, value in enumerate(y_axis) if index % 5 == 0]
    #
    # plt.figure()
    # cm = plt.cm.get_cmap('viridis')
    # sns.lineplot(x_axis, y_axis)
    # sns.set()
    #
    # plt.title('Average rewards per number of episodes')
    # plt.show()
    #
    # len(gym.envs.registry.all())
    # list(map(lambda x: 'CartPole' in x.id, list(gym.envs.registry.all())))

    # perturbation = 1

    # total_rewards = [play_policy(p, perturbation) for p in POLICIES]

for i in range(10 if False else 3, 10 + 1, 4):
    print(i)
