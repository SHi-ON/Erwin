from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.stats import iqr
from sklearn.cluster import KMeans

from rlessential.domains import MachineReplacementMDP
from agents import MachineReplacementMDPAgent
from rlessential.solvers import ValueIteration
from util import RANDOM_SEED
from consts import *


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
    bin_width = 2 * iq_range / (len(samples) ** (1 / 3))

    hi = np.max(samples, axis=0)
    lo = np.min(samples, axis=0)

    bin_count = (hi - lo) / bin_width
    bin_count = bin_count.astype(int)
    bin_count = bin_count.item()

    return bin_count


def cluster_values(values, num_buckets, random_seed):
    kmeans = KMeans(n_clusters=num_buckets, random_state=random_seed)
    kmeans.fit(values)

    return kmeans.labels_


def aggregate(mdp, agg_map):

    # original rewards
    rewards_orig = dict(zip(domain_mr.mdp[COL_STATE_TO], domain_mr.mdp[COL_REWARD]))

    agg_state = list(set(agg_map.values()))

    # aggregate rewards
    rewards_agg = defaultdict(int)
    for k, v in agg_map.items():
        rewards_agg[v] += rewards_orig[k]

    # get a deep copy
    df_agg = mdp.copy(deep=True)

    # state mapping
    # should be done only once!
    df_agg.loc[:, [COL_STATE_FROM, COL_STATE_TO]] = \
        df_agg.loc[:, [COL_STATE_FROM, COL_STATE_TO]].replace(agg_map)

    # reward mapping
    for k, v in rewards_agg.items():
        df_agg.loc[df_agg[COL_STATE_TO] == k, COL_REWARD] = v

    # probability mapping
    for s in agg_state:
        for a in range(domain_mr.num_actions):
            for sp in agg_state:
                cond = ((df_agg[COL_STATE_FROM] == s) & (df_agg[COL_ACTION] == a) & (df_agg[COL_STATE_TO] == sp))
                df = df_agg.loc[cond]
                if len(df) != 0:
                    prob = df[COL_PROBABILITY].mean()
                    df_agg.loc[cond, COL_PROBABILITY] = prob

    df_agg.drop_duplicates(inplace=True)
    df_agg.sort_values(by=[COL_STATE_FROM, COL_ACTION, COL_STATE_TO], inplace=True)
    df_agg.reset_index(drop=True, inplace=True)

    return df_agg


if __name__ == '__main__':
    mdp_input = pd.read_csv('dataset/mdp/machine_replacement_mdp.csv')

    gamma = 0.90
    epsilon = 0.0000001
    tau = (epsilon * (1 - gamma)) / (2 * gamma)
    steps = 5000

    domain_mr = MachineReplacementMDP(mdp_input)
    solver_vi = ValueIteration(domain_mr, discount=gamma, threshold=tau, verbose=True)
    agent_mr = MachineReplacementMDPAgent(domain_mr, solver_vi, discount=gamma, horizon=steps)

    agent_mr.train()
    values = solver_vi.get_v_table()

    agent_mr.run(policy=None, randomized=True)
    total_reward_mr = agent_mr.total_reward
    samples = agent_mr.samples

    samples = preprocess_samples(samples)  # extract states from the samples
    num_buckets = discretize_samples(samples)  # range calculation

    agg_values_labels = cluster_values(values, num_buckets, RANDOM_SEED)

    # mapping
    aggregate_map = dict(enumerate(agg_values_labels))

    # reverse dictionary with duplicates
    agg_to_orig = dict()
    for s, s_agg in aggregate_map.items():
        agg_to_orig.setdefault(s_agg, list()).append(s)

    # synthesize the aggregate mdp
    agg_mdp = aggregate(domain_mr.mdp, aggregate_map)

    domain_agg_mr = MachineReplacementMDP(agg_mdp)
    solver_agg_vi = ValueIteration(domain_agg_mr, discount=gamma, threshold=tau, verbose=True)
    agent_agg_mr = MachineReplacementMDPAgent(domain_agg_mr, solver_agg_vi, discount=gamma, horizon=steps)

    agent_agg_mr.train()
    agg_values = solver_agg_vi.get_v_table()
    agg_policy = solver_agg_vi.calculate_policy()

    agent_agg_mr.run(policy=None, randomized=False)
    total_reward_agg_mr = agent_agg_mr.total_reward

    # TODO: run aggregate policy on the true model (original)
    orig_agg_policy = np.zeros((domain_mr.num_states, 1))
    for s_agg, s_group in agg_to_orig.items():
        for s in s_group:
            orig_agg_policy[s] = agg_policy[s_agg]

    agent_mr.run(policy=orig_agg_policy, randomized=False)
    total_reward_mr_true = agent_mr.total_reward

    print('original reward:', total_reward_mr)
    print('aggregate reward:', total_reward_agg_mr)
    print('true aggregate reward:', total_reward_mr_true)



