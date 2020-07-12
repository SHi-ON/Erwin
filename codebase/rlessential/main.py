import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import iqr
from sklearn.cluster import KMeans

import consts
from agents import BaseAgent
from rlessential.domains import MachineReplacementMDPDomain, RiverSwimMDPDomain, SixArmsMDPDomain
from rlessential.solvers import ValueIterationSolver

# hyper-parameters
gamma = 0.90
epsilon = 0.0001
tau = (epsilon * (1 - gamma)) / (2 * gamma)

steps = 3000


def average_rolling(array):
    """
    Calculates moving(rolling) average.

    :param array: input array
    :return: calculated cumulative moving average
    """
    cumulative_sum = array.cumsum()
    a = np.arange(array.size)
    a = a + 1
    rolling_mean = cumulative_sum / a
    return rolling_mean


def extract_states(samples):
    """
    Extracts states from the samples. The function preserves duplicates.

    :param samples: collected samples
    :return: states
    """
    states = [sample_.current_state for sample_ in samples]
    states = np.stack(states, axis=0)

    assert len(states) == len(samples)
    assert states.shape == (len(samples), 1)

    return states


def calculate_bin_width(samples):
    inter_quartile_range = iqr(samples, axis=0)
    bin_width = 2 * inter_quartile_range / (len(samples) ** (1 / 3))
    return bin_width


def calculate_bin_count(samples, bin_width):
    hi = np.max(samples, axis=0)
    lo = np.min(samples, axis=0)

    bin_count = (hi - lo) / bin_width
    bin_count = np.ceil(bin_count)
    bin_count = bin_count.astype(int)
    bin_count = bin_count.item()
    return bin_count


def cluster_values(samples, bin_count):
    kmeans = KMeans(n_clusters=bin_count,
                    random_state=consts.RANDOM_SEED)
    kmeans.fit(samples)
    labels = kmeans.labels_
    return labels


def map_aggregate_states(labels):
    """
    Associates an original state to an aggregate state (aggregation), and vice versa (disaggregation).

    :param labels: clustering labels
    :return: tuple of both aggregation and disaggregation mappings
    """
    aggregation_mapping = dict(enumerate(labels))

    disaggregation_mapping = dict()
    for s_original, s_aggregate in aggregation_mapping.items():
        disaggregation_mapping.setdefault(s_aggregate, list()).append(s_original)

    return aggregation_mapping, disaggregation_mapping


def map_aggregate_rewards(state_mapping, original_domain):
    """
    Associates original and aggregate rewards with the corresponding ending state.

    :param state_mapping: state aggregation mapping
    :param original_domain: the original domain
    :return: tuple of both original and aggregate rewards
    """
    original_rewards = dict(zip(original_domain.mdp[consts.COL_STATE_TO], original_domain.mdp[consts.COL_REWARD]))

    aggregate_rewards = dict()
    for s_original, s_aggregate in state_mapping.items():
        aggregate_rewards.setdefault(s_aggregate, int())
        aggregate_rewards[s_aggregate] += original_rewards[s_original]

    return original_rewards, aggregate_rewards


def map_aggregate_policy(aggregate_policy, state_mapping, original_domain):
    """
    Adapts a policy compatible with the original agent from the given aggregate policy.
    
    :param aggregate_policy: aggregate policy
    :param state_mapping: state aggregation mapping
    :param original_domain: the original domain 
    :return: the policy
    """
    aggregate_policy_original = np.zeros((original_domain.num_states, 1))
    for s_original, s_aggregate in state_mapping.items():
        aggregate_policy_original[s_original] = aggregate_policy[s_aggregate]

    return aggregate_policy_original


def aggregate_mdp(values, bin_count, domain):
    clustered_state_labels = cluster_values(values, bin_count)

    df = domain.mdp.copy(deep=True)

    # state mapping
    aggregation_states, _ = map_aggregate_states(clustered_state_labels)
    state_columns = [consts.COL_STATE_FROM, consts.COL_STATE_TO]
    df.loc[:, state_columns] = df.loc[:, state_columns].replace(aggregation_states)

    # reward mapping
    _, aggregation_rewards = map_aggregate_rewards(aggregation_states, domain)
    for s_original, r_aggregate in aggregation_rewards.items():
        reward_condition = df[consts.COL_STATE_TO] == s_original
        df.loc[reward_condition, consts.COL_REWARD] = r_aggregate

    # transition probability mapping
    for s in aggregation_states.keys():
        for a in range(domain.num_actions):
            for sp in aggregation_states.keys():
                transition_condition = ((df[consts.COL_STATE_FROM] == s) &
                                        (df[consts.COL_ACTION] == a) &
                                        (df[consts.COL_STATE_TO] == sp))
                df_transition = df.loc[transition_condition]
                if len(df) != 0:
                    transition_probability = df_transition[consts.COL_PROBABILITY].mean()
                    df.loc[transition_condition, consts.COL_PROBABILITY] = transition_probability

    df.drop_duplicates(inplace=True)
    df.sort_values(by=[consts.COL_STATE_FROM, consts.COL_ACTION, consts.COL_STATE_TO], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, aggregation_states


def run(mdp_domain):
    domain = mdp_domain()
    solver = ValueIterationSolver(domain,
                                  discount=gamma,
                                  threshold=tau,
                                  verbose=True)
    agent = BaseAgent(domain,
                      solver,
                      epochs=steps)
    state_values = agent.train()
    rewards, samples = agent.run(external_policy='randomized')

    states = extract_states(samples)
    bucket_width = calculate_bin_width(samples=states)
    bucket_count = calculate_bin_count(samples=states, bin_width=bucket_width)

    mdp_aggregate, aggregation_mapping = aggregate_mdp(values=state_values,
                                                       bin_count=bucket_count,
                                                       domain=domain)

    domain_aggregate = mdp_domain(mdp_aggregate)
    solver_aggregate = ValueIterationSolver(domain=domain_aggregate,
                                            discount=gamma,
                                            threshold=tau,
                                            verbose=True)
    agent_aggregate = BaseAgent(domain=domain_aggregate,
                                solver=solver_aggregate,
                                epochs=steps)
    state_values_aggregate = agent_aggregate.train()
    rewards_aggregate, samples_aggregate = agent_aggregate.run()
    policy_aggregate = solver_aggregate.policy

    adapted_policy_aggregate = map_aggregate_policy(aggregate_policy=policy_aggregate,
                                                    state_mapping=aggregation_mapping,
                                                    original_domain=domain)
    domain.reset()
    rewards_aggregate_adapted, samples_aggregate_adapted = agent.run(external_policy=adapted_policy_aggregate)

    print('original return:', rewards.sum())
    print('aggregate return:', rewards_aggregate.sum())
    print('adapted return:', rewards_aggregate_adapted.sum())

    return rewards, rewards_aggregate, rewards_aggregate_adapted


def visualize_rewards(rewards, rewards_aggregate, rewards_aggregate_adapted):
    titles = {'original': 'Original rewards', 'aggregate': 'Aggregate rewards', 'adapted': 'Adapted aggregate rewards'}

    def plot_steps(original, aggregate, adapted):
        fig, ax = plt.subplots()
        ax.set_xticks([], [])
        ax.set_yticks([], [])

        ax1 = fig.add_subplot(311)
        ax1.plot(original, color='r', label=titles['original'])
        ax1.set_title(titles['original'])
        ax1.set_xticks([], [])

        ax2 = fig.add_subplot(312)
        ax2.plot(aggregate, color='b', label=titles['aggregate'])
        ax2.set_title(titles['aggregate'])
        ax2.set_xticks([], [])
        ax2.set_ylabel('Rewards')

        ax3 = fig.add_subplot(313)
        ax3.plot(adapted, color='g', label=titles['adapted'])
        ax3.set_title(titles['adapted'])

        plt.xlabel('Steps')
        return plt

    def plot_cumulative(original, aggregate, adapted):
        plt.figure()
        plt.plot(original.cumsum(),
                 color='r', label=titles['original'])
        plt.plot(aggregate.cumsum(),
                 color='b', label=titles['aggregate'])
        plt.plot(adapted.cumsum(),
                 color='g', label=titles['adapted'])
        plt.xlabel('Steps')
        plt.ylabel('Cumulative reward')
        plt.legend()
        return plt

    def plot_rolling(original, aggregate, adapted):
        rolling_original = average_rolling(original)
        rolling_aggregate = average_rolling(aggregate)
        rolling_adapted = average_rolling(adapted)

        plt.figure()
        plt.plot(rolling_original, color='r', label=titles['original'])
        plt.plot(rolling_aggregate, color='b', label=titles['aggregate'])
        plt.plot(rolling_adapted, color='g', label=titles['adapted'])
        plt.xlabel('Steps')
        plt.ylabel('Rolling average reward')
        plt.legend()
        return plt

    plot_steps(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.show()

    plot_cumulative(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.show()

    plot_rolling(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.show()


if __name__ == '__main__':
    machine_replacement_domain = MachineReplacementMDPDomain
    river_swim_domain = RiverSwimMDPDomain
    six_arms_domain = SixArmsMDPDomain
    run(machine_replacement_domain)
