import time

import gym
import numpy as np
import pandas as pd
import tqdm

# number of runs to determine how good is the policy
trials = 2

env = gym.make('CartPole-v1')
policy = pd.read_csv("policy_nn.csv")
# converting to an nparray
scales = pd.read_csv("scales.csv").values

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

totalreward = 0

# n = 0
# m = 0
info_list = []

# will not include sleep time as time.perf_counter will
t_perf = time.perf_counter()
t_elapsed = time.process_time()

eps = 0.09

for trial in tqdm.trange(trials):
    env.reset()
    done = False
    for i in range(200):
        env.render()
        time.sleep(0.05)
        if i > 0:
            # find the closest state
            statescaled = state @ scales
            statefeatures += eps * state
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
                action = int(actions[statei])

            print(i, stateids[statei], action, values[statei], np.linalg.norm(statescaled - statefeatures[statei]))
        else:
            # n += 1
            action = env.action_space.sample()

        # stop only after saving the state
        if done:
            break

        [state, reward, done, info] = env.step(action)  # take a random action
        print(state, reward, done, info)
        env.render()


        # (state, reward, done, info) = en v.step(action)  # take a random action
        if any(info):
            info_list.append(info)

        totalreward += reward

env.close()

elapsed_time = time.process_time() - t_elapsed
perf_time = time.perf_counter() - t_perf

print("Average reward per trial", totalreward / trials)

print("elapsed time:", elapsed_time)
print("elapsed time:", perf_time)
print('list of info is:', info_list)


