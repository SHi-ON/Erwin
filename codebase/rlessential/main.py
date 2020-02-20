import pickle

import numpy as np
from scipy.stats import iqr

from agents import QLearningCartPoleAgent, QLearningMountainCarAgent, CartPoleAgent, RAAMAgent


def q_learning():
    # discounted-reward problem
    agent = QLearningCartPoleAgent(num_buckets=(1, 2, 6, 12), num_episodes=1000, discount=0.99)
    # agent = QLearningMountainCarAgent(num_buckets=(6, 6), num_episodes=10000, discount=0.99)

    # training the agent to learn the baseline Q-table
    agent.train()

    # agent.run(False)

    # simulation to generate samples by the baseline policy
    agent.simulate()
    samples = agent.batch

    sample_size = len(samples)
    print('Sample size:', sample_size)

    avg_spe = (sample_size / 1000)
    print('Average step per episode:', avg_spe)

    pickle.dump(samples, open('samples.p', 'wb'))


def robust_learning():
    agent = CartPoleAgent()
    agent.run()
    samples = agent.samples
    samples = np.array(samples)
    print(samples.shape)
    env = agent.env

    raam = RAAMAgent(samples=samples, env=env)
    raam.aggregate_states()
    len(raam.S)
    raam.A.shape
    raam.B.shape


def cartpole_iqr_compare():
    print("loading samples")

    samples = pickle.load(open('samples.p', 'rb'))

    print("unifying samples")

    ss = []
    for s in samples:
        ss.append(s[0])
        # ss.append(s[3])
    len(samples)
    len(ss)

    print("stacking")

    smp = np.stack(ss, axis=0)
    smp.shape

    n = len(smp)
    iq_range = iqr(smp, axis=0)
    h = 2 * iq_range * (n ** (-1 / 3))

    hi = np.max(smp, axis=0)
    lo = np.min(smp, axis=0)

    counts = (hi - lo) / h
    counts = counts / 1
    counts = np.round(counts)
    counts = counts.astype(int)
    num_buckets = tuple(counts)

    print("agents")

    agent_base = QLearningCartPoleAgent(num_buckets=(1, 2, 6, 12), num_episodes=1000, discount=0.99)
    agent_base.train()
    agent_base.run(False)

    agent_hist = QLearningCartPoleAgent(num_buckets=num_buckets, num_episodes=1000, discount=0.99)
    agent_hist.train()
    agent_hist.run(False)


if __name__ == '__main__':
    # q_learning()

    # robust_learning()

    # cartpole_iqr_compare()



    pass



