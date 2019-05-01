import time

import gym
import numpy as np
import pandas as pd
import tqdm

import matplotlib.pyplot as plt


def select_action(s):
    state_q = q_values.loc[q_values['idstate'] == s]
    a = state_q.max()[1]
    return int(a)


env = gym.make('CartPole-v1')

# policy = pd.read_csv("initial/policy_nn.csv")
policy = pd.read_csv("qvals/policy_nn.csv")

q_values = pd.read_csv("qvals/qvalues.csv")

# scales = pd.read_csv("initial/scales.csv").values # converting to an ndarray
scales = pd.read_csv("qvals/scales.csv").values  # converting to an ndarray

# policy = pd.read_csv("codebase/adrel/marek/policy_nn.csv")
# scales = pd.read_csv("codebase/adrel/marek/scales.csv").values

statefeatures = policy.values[:, 0:4]
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
trials = 100
is_q = True
total_reward = 0

# n = 0
# m = 0
rewards = dict()
info_list = []

# will not include sleep time as time.perf_counter will
t_perf = time.perf_counter()
t_elapsed = time.process_time()

eps = 0.09

for trial in tqdm.trange(trials):
    env.reset()
    done = False
    for i in range(200):
        # env.render()
        # time.sleep(0.05)
        if i > 0:
            # find the closest state
            statescaled = state @ scales
            # statefeatures += eps * state
            dst = np.linalg.norm(statefeatures - np.repeat(np.atleast_2d(statescaled), statefeatures.shape[0], 0),
                                 axis=1)

            statei = np.argmin(dst)  # state index in the file, not the number of the state

            if policy_randomized:
                idstate = stateids[statei]
                all_statei = np.where(stateids == idstate)[0]  # find all relevant state ids
                all_probs = probabilities[all_statei]
                all_acts = actions[all_statei]
                assert (abs(1 - sum(all_probs)) < 0.01)
                action = int(np.random.choice(all_acts, p=all_probs))
                # m += 1
            else:
                # assume that there is a single action for each state
                # switch between q-learning or the policy
                action = select_action(statei) if is_q else int(actions[statei])

            # print(i, stateids[statei], action, values[statei], np.linalg.norm(statescaled - statefeatures[statei]))
        else:
            # n += 1
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

print("Average reward per trial", total_reward / trials)

print("elapsed time:", elapsed_time)
print("elapsed time:", perf_time)
print('list of info is:', info_list)

x_axis = np.array([i for i in range(trials)])
y_axis = np.array([i for i in rewards.values()])

cm = plt.cm.get_cmap('viridis')
# cc = np.linspace(0, 2, 30)
# sc = plt.scatter(x_axis, y_axis, vmin=0, vmax=200)
plt.plot(x_axis, y_axis, '-')
# plt.colorbar(sc)
plt.title("Reward per each trial using the provided {}".format('q values' if is_q else 'policy'))
plt.show()


