import random
import gym
env = gym.make('CartPole-v1')
random.seed(2018)

laststate = None
samples = []
for k in range(20):
  env.reset()
  done = False
  for i in range(100):
    #env.render()
    action = env.action_space.sample()
    if i > 0:
      samples.append( (i-1,) + tuple(state) + (action,) + (reward,) )
    # stop only after saving the state
    if done:
      break
    [state,reward,done,info] = env.step(action) # take a random action
env.close()