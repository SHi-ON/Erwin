"""
Adversarial RL framework

**Notes:

- ranges all START from 1 to STOP + 1
"""
import time

import numpy as np
import gym
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from agent_policy import AgentPolicy
from agent_dqn import AgentDQN
from util import timing, pickle_store

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

""" CartPole physics parameters:

    self.masspole = 0.1
    self.masscart = 1.0 **
    self.length = 0.5  # actually half the pole's length **

    self.polemass_length = (self.masspole * self.length)
    self.total_mass = (self.masspole + self.masscart)

    self.gravity = 9.8
    self.force_mag = 10.0
    self.tau = 0.02  # seconds between state updates
    self.kinematics_integrator = 'euler'
"""

# gym/envs/__init__.py'
# different max_episode_steps 'CartPole-v0' and 'CartPole-v1'
GYM_ENV = 'CartPole-v1'
CONTROL_ALG = 'DQN'

SEQ_RUN = True
IS_LOAD = False
IS_RENDER = False

EPISODES = 1000
SEQ_STEPS = 25
PHYS_EPISODES = 300
PHYS_SEQ_STEPS = 25

EPISODE_STEPS = 200
BATCH_SIZE = 32
PENALTY = -10

POLICIES = [{'dir': 'initial', 'q': False},
            {'dir': 'qvals', 'q': True}]


def play_policy(policy_info, pert_noise, num_episodes):
    env = gym.make(GYM_ENV)
    agent = AgentPolicy(policy_info)

    game_reward = dict()
    for episode in tqdm.trange():
        env.reset()
        # take an initial random action and observe the state
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if done:
            continue
        for step in range(1, num_episodes):
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


def play_dqn(env, num_episodes, file_name=None, load=False):
    space_size_state = env.observation_space.shape[0]
    space_size_action = env.action_space.n
    agent = AgentDQN(space_size_state, space_size_action)
    if (file_name is not None) and (load is not False):
        agent.load(file_name + '.h5')
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
        if (file_name is not None) and (episode % 10 == 0):
            agent.save(file_name + '.h5')
    env.close()
    return _game_rewards


if __name__ == "__main__":
    print('***** DQN training...')

    rewards_total = dict()
    env_stock = gym.make(GYM_ENV)
    name_env = env_stock.unwrapped.spec.id
    file_name_stock = 'outputs/{0}_{1}_batchsize-{2}_episodes-{3}_step-{4}'.format(CONTROL_ALG, name_env,
                                                                                   BATCH_SIZE, EPISODES,
                                                                                   SEQ_STEPS)
    t_start = time.perf_counter()
    for seq_ep in range(SEQ_STEPS if SEQ_RUN else EPISODES, EPISODES + 1, SEQ_STEPS):
        game_rewards = play_dqn(env_stock, seq_ep, file_name_stock, load=IS_LOAD)
        game_reward_total = sum(game_rewards.values())
        game_reward_avg = game_reward_total / seq_ep
        print("\t Average reward per episode {:.2f}".format(game_reward_avg))
        rewards_total[seq_ep] = game_reward_total
    print('\n', '-' * 10, 'Summary', '-' * 10)
    t_finish = timing(t_start)
    sum_rewards_total = sum(rewards_total.values())
    avg_rewards_total = sum_rewards_total / sum(rewards_total.keys())
    time_per_step = t_finish / sum_rewards_total
    print('*** Time per step: {:.4f} second(s)'.format(time_per_step))
    print('*** Sum of total rewards: {}'.format(sum_rewards_total))
    print('*** Average total reward: {:.2f}'.format(avg_rewards_total))

    '''--------------------Plotting--------------------'''
    sequence_episode = np.array(list(rewards_total.keys()))
    sequence_reward = np.array(list(rewards_total.values()))

    plt.figure()
    cm = plt.cm.get_cmap('viridis')
    sns.lineplot(sequence_episode, sequence_reward)
    sns.set()
    plt.title('Average reward per number of episodes')
    plt.xlabel('Average reward')
    plt.ylabel('Episode')
    plt.savefig(file_name_stock + '.png')
    plt.show()

    ''' ====================Physics==================== '''
    print('\n', '=' * 30, '\n')
    print('***** Customized physics...')

    masses_cart = np.array([0.1, 0.5, 1.0, 1.5, 2])
    lengths_pole = np.array([0.05, 0.25, 0.5, 0.75])
    phys_return = np.zeros((len(masses_cart), len(lengths_pole)))

    env_custom = gym.make(GYM_ENV)
    name_env = env_custom.unwrapped.spec.id
    file_name_custom = 'outputs/{0}_{1}-custom_batchsize-{2}_episodes-{3}_step-{4}_{5}x{6}'.format(CONTROL_ALG,
                                                                                                   name_env,
                                                                                                   BATCH_SIZE,
                                                                                                   PHYS_EPISODES,
                                                                                                   PHYS_SEQ_STEPS,
                                                                                                   len(masses_cart),
                                                                                                   len(lengths_pole))
    t_start = time.perf_counter()
    for i, mc in enumerate(masses_cart):
        for j, lp in enumerate(lengths_pole):
            print('** \t cart mass:{0} - pole length:{1}'.format(mc, lp))
            env_custom.masscart = mc
            env_custom.length = lp
            phys_rewards_total = dict()
            for phys_seq_ep in range(PHYS_SEQ_STEPS if SEQ_RUN else PHYS_EPISODES, PHYS_EPISODES + 1, PHYS_SEQ_STEPS):
                phys_game_rewards = play_dqn(env_custom, phys_seq_ep)
                phys_game_reward_total = sum(phys_game_rewards.values())
                phys_game_reward_avg = phys_game_reward_total / phys_seq_ep
                print("\t Average reward per episode {:.2f}".format(phys_game_reward_avg))
            phys_return[i][j] = round(phys_game_reward_total / phys_seq_ep, ndigits=3)
    t_finish = timing(t_start)

    '''--------------------Plotting--------------------'''
    sns.heatmap(phys_return, annot=True, fmt='.2f', xticklabels=lengths_pole, yticklabels=masses_cart)
    plt.title('Average rewards per number of episodes')
    plt.xlabel('Pole length')
    plt.ylabel('Cart mass')
    plt.savefig(file_name_custom + '.png')
    plt.show()

    ''' ====================end==================== '''
    # the whole OpenAI Gym environments
    len(gym.envs.registry.all())
    is_EnvSpec = list(map(lambda x: 'CartPole' in x.id, list(gym.envs.registry.all())))
    any(is_EnvSpec)

    # perturbation = 1
    # total_rewards = [play_policy(p, perturbation) for p in POLICIES]
