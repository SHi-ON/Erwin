import time

import gym
import numpy as np
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


"""
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
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """


def select_action(s):
    state_q = q_values.loc[q_values['idstate'] == s]
    a = state_q.loc[state_q['qvalue'] == state_q.max()[2]]['idaction']
    return int(a)


env = gym.make('CartPole-v1')

# policy = pd.read_csv("initial/policy_nn.csv")
policy = pd.read_csv("qvals/policy_nn.csv")

q_values = pd.read_csv("qvals/qvalues.csv")

# scales = pd.read_csv("initial/scales.csv").values # converting to an ndarray
scales = pd.read_csv("qvals/scales.csv").values  # converting to an ndarray

# policy = pd.read_csv("codebase/adrel/marek/policy_nn.csv")
# scales = pd.read_csv("codebase/adrel/marek/scales.csv").values

state_features = policy.values[:, 0:4]
stateids = np.array(policy["State"])
actions = np.array(policy["Action"])
values = np.array(policy["Value"])

if "Probability" in policy.columns:
    print("Using a randomized policy")
    policy_randomized = True
    probabilities = np.array(policy["Probability"])
else:
    print("Using a deterministic policy")
    policy_randomized = False
    probabilities = None

# number of runs to determine how good is the policy
trials = 1000
eps = 5
is_q = True

rewards = dict()
info_list = []
total_reward = 0

# will not include sleep time as time.perf_counter will
t_perf = time.perf_counter()
t_elapsed = time.process_time()

for trial in tqdm.trange(trials):
    env.reset()
    done = False
    for i in range(200):
        # faster with no graphical output
        # env.render()
        # time.sleep(0.05)
        if i > 0:
            # scale observed state
            state_scaled = state @ scales
            state_scaled = state_scaled @ np.eye(scales.shape[0]) * (1 + eps)
            # ||Sc+e - Sc||
            dst = np.linalg.norm(state_features - np.repeat(np.atleast_2d(state_scaled), state_features.shape[0], 0),
                                 axis=1)
            # find the closest state to the observed state in the policy
            statei = np.argmin(dst)  # state index in the file, not the number of the state

            if policy_randomized:
                idstate = stateids[statei]
                all_statei = np.where(stateids == idstate)[0]  # find all relevant state ids
                all_probs = probabilities[all_statei]
                all_acts = actions[all_statei]
                assert (abs(1 - sum(all_probs)) < 0.01)
                action = int(np.random.choice(all_acts, p=all_probs))
            else:
                # assume that there is a single action for each state
                # switch between q-learning or the policy
                action = select_action(statei) if is_q else int(actions[statei])

            # print(i, stateids[statei], action, values[statei], np.linalg.norm(state_scaled - state_features[statei]))
        else:
            action = env.action_space.sample()

        # stop only after saving the state
        if done:
            break

        [state, reward, done, info] = env.step(action)  # take a random action
        # print(state, reward, done, info)
        # env.render()

        # (state, reward, done, info) = en v.step(action)  # take a random action
        if any(info):
            info_list.append(info)

        if trial in rewards:
            rewards[trial] += reward
        else:
            rewards[trial] = reward

        total_reward += reward

env.close()

elapsed_time = time.process_time() - t_elapsed
perf_time = time.perf_counter() - t_perf

average_reward = total_reward / trials
print("Average reward per trial", average_reward)

print("elapsed time:", elapsed_time)
print("elapsed time:", perf_time)
print('list of info is:', info_list)

x_axis = np.array([i for i in range(trials)])
y_axis = np.array([i for i in rewards.values()])

cm = plt.cm.get_cmap('viridis')
# cc = np.linspace(0, 2, 30)
# sc = plt.scatter(x_axis, y_axis, vmin=0, vmax=200)
# plt.plot(x_axis, y_axis, '-')
# plt.ylim((0, 200))
# plt.yticks([1, 100, 200])

plt.figure()
sns.lineplot(x_axis, y_axis)
sns.set()

plt.title("Reward per each episode | {} | eps={} | avg. reward={}"
          .format('q_values' if is_q else 'policy', eps, average_reward))
plt.show()
