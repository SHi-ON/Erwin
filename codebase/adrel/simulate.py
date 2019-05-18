import numpy as np
import tqdm

import gym

import matplotlib.pyplot as plt
import seaborn as sns

from agent_policy import AgentPolicy
from agent_dqn import AgentDQN

# gym/envs/__init__.py'
# different max_episode_steps 'CartPole-v0' and 'CartPole-v1'
GYM_ENV = 'CartPole-v1'
EPISODES = 1000
MAX_EPISODE_STEPS = 200
BATCH_SIZE = 32


IS_RENDER = False

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


def play_dqn():
    env = gym.make(GYM_ENV)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = AgentDQN(state_size, action_size)
    # agent.load('./model/cartpole-dqn.h5')

    game_reward = dict()
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for step in range(MAX_EPISODE_STEPS):
            if IS_RENDER:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(episode, EPISODES, step, agent.epsilon))
                game_reward[episode] = step + 1
                break
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
            if step == MAX_EPISODE_STEPS - 1:
                game_reward[episode] = step + 1

        # if episode % 10 == 0:
        #     agent.save("./model/cartpole-dqn.h5")
    env.close()
    return game_reward


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
        for step in range(1, MAX_EPISODE_STEPS):
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
            if step == MAX_EPISODE_STEPS - 1:
                game_reward[episode] = step + 1
    env.close()
    return game_reward


if __name__ == "__main__":

    perturbation = 1

    total_rewards = [play_policy(p, perturbation) for p in POLICIES]
    total_rewards.append(play_dqn())

    average_reward = sum(total_rewards.values()) / EPISODES
    print("Average reward per trial", average_reward)

    x_axis = np.array([i for i in range(EPISODES)])
    y_axis = np.array([i for i in total_rewards.values()])

    cm = plt.cm.get_cmap('viridis')
    plt.figure()
    sns.lineplot(x_axis, y_axis)
    sns.set()

    plt.title("Reward per episode | {} | eps={} | avg. reward={}"
              .format(p['dir'], epsilon, average_reward))
    plt.show()


