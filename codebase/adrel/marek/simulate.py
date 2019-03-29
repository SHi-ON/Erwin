import gym
import tqdm
import pandas as pd
import numpy as np
import time

# number of runs to determine how good is the policy
trials = 2

env = gym.make('CartPole-v1')
policy = pd.read_csv("policy_nn.csv")
scales = pd.read_csv("scales.csv").values

# policy = pd.read_csv("codebase/adrel/marek/policy_nn.csv")
# scales = pd.read_csv("codebase/adrel/marek/scales.csv").values

statefeatures = policy.values[:,0:4]
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

for trial in tqdm.trange(trials):
    env.reset()
    done = False
    for i in range(200):
        env.render()
        time.sleep(0.05)
        if i > 0:
            # find the closest state
            statescaled = state @ scales
            statefeatures += 1
            dst = np.linalg.norm(statefeatures - np.repeat(np.atleast_2d(statescaled), statefeatures.shape[0], 0), axis=1)    
            
            statei = np.argmin(dst) # state index in the file, not the number of the state
            
            if policy_randomized:
                idstate = stateids[statei]
                all_statei = np.where(stateids == idstate)[0] # find all relevant state ids
                all_probs = probabilities[all_statei]
                all_acts = actions[all_statei]
                assert(abs(1-sum(all_probs)) < 0.01)
                action = int(np.random.choice(all_acts, p = all_probs))
            else:
                # assume that there is a single action for each state
                action = int(actions[statei])
            
            print(i, stateids[statei], action, values[statei], np.linalg.norm(statescaled - statefeatures[statei]) )
        else:
            action = env.action_space.sample()
        
        # stop only after saving the state
        if done:
            break
        
        [state,reward,done,info] = env.step(action) # take a random action
        totalreward += reward

env.close()
print("Average reward per trial", totalreward / trials)
