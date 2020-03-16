
import pandas as pd
import numpy as np
from scipy.stats import iqr
from sklearn.cluster import KMeans

from rlessential.domains import MachineReplacementMDP
from agents import MachineReplacementMDPAgent
from rlessential.solvers import ValueIteration
from util import RANDOM_SEED


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


def preprocess_samples(samples, verbose=True):
    state_samples = [sample_.current_state for sample_ in samples]
    state_samples = np.stack(state_samples, axis=0)

    if verbose:
        print('{} state samples are processed.'.format(len(state_samples)))
        print('output samples shape: {}'.format(state_samples.shape))

    return state_samples


def discretize_samples(samples):
    iq_range = iqr(samples, axis=0)
    bin_width = 2 * iq_range * (len(samples) ** (-1 / 3))

    hi = np.max(samples, axis=0)
    lo = np.min(samples, axis=0)

    bin_count = (hi - lo) / bin_width
    bin_count = np.round(bin_count).astype(int)
    bin_count = bin_count.item()

    return bin_count


def discretize_values(values):
    pass


if __name__ == '__main__':

    mdp = pd.read_csv('dataset/mdp/machine_replacement_mdp.csv')

    gamma = 0.90
    epsilon = 0.0000001
    tau = (epsilon * (1 - gamma)) / (2 * gamma)
    steps = 1000

    domain_mr = MachineReplacementMDP(mdp)
    solver_vi = ValueIteration(domain_mr, discount=gamma, threshold=tau, verbose=True)
    agent_mr = MachineReplacementMDPAgent(domain_mr, solver_vi, discount=gamma, horizon=steps)

    agent_mr.train()
    values = solver_vi.get_v_table()

    agent_mr.run()
    samples = agent_mr.samples
    samples = preprocess_samples(samples)  # extract states from the samples
    num_buckets = discretize_samples(samples)  # range calculation

    kmeans = KMeans(n_clusters=num_buckets, random_state=RANDOM_SEED)
    kmeans.fit(values)
    kmeans.labels_



