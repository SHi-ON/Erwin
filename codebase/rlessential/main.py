
import pandas as pd
import numpy as np

from rlessential.domains import MachineReplacementMDP
from agents import MachineReplacementAgent
from rlessential.solvers import ValueIteration


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

    mdp = pd.read_csv('dataset/mdp/machine_replacement_mdp.csv')

    domain_mr = MachineReplacementMDP(mdp)
    agent_mr = MachineReplacementAgent(domain_mr)


    gamma = 0.90
    epsilon = 0.0000001
    tau = (epsilon * (1 - gamma)) / (2 * gamma)

    # even different initial values will end up with the same state values!
    init_val = np.arange(10)
    init_val = np.random.rand(10) * 10

    vi = ValueIteration(domain_mr, discount=gamma, initial_values=init_val, threshold=tau, verbose=True)
    vi.calculate_value()




