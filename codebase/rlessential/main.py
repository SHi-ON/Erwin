import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import iqr, skew
from sklearn.cluster import KMeans

import consts
from agents import BaseAgent
from rlessential.domains import MachineReplacementMDPDomain, RiverSwimMDPDomain, SixArmsMDPDomain
from rlessential.solvers import ValueIterationSolver

# hyper-parameters
GAMMA = 0.90
EPSILON = 0.0001
TAU = (EPSILON * (1 - GAMMA)) / (2 * GAMMA)

STEPS = 5000

# ['scott', 'fd', 'ss']
WIDTH_METHOD = 'ss'
# ['base', 'sqrt', 'sturge', 'rice', 'doane']
COUNT_METHOD = 'base'


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
    def scott(array):
        width = 3.49 * array.std() / (len(array) ** (1 / 3))
        return width

    def freedman_diaconis(array):
        inter_quartile_range = iqr(array, axis=0)
        width = 2 * inter_quartile_range / (len(array) ** (1 / 3))
        width = width.item()
        return width

    def shimazaki_shinomoto(array):
        """
        Shimazaki and Shinomoto's choice. Biased variance is recommended.

        :param array: sample array
        :type array: np.ndarray

        :return: bin-width
        :rtype: float
        """

        def l2_risk_function(points):
            lo = points.min()
            hi = points.max()

            n_min = lo + 1  # min number of splits
            n_max = hi + 1  # max number of splits

            counts = np.arange(n_min, n_max)  # number of bins vector
            widths = (hi - lo) / counts  # width (size) of bins vector

            costs = np.zeros((widths.size, 1))  # cost function values
            for i in range(counts.size):
                edges = np.linspace(lo, hi, counts[i] + 1)
                current_frequency = plt.hist(points, edges)  # number of data points in the current bin
                current_frequency = current_frequency[0]
                costs[i] = (2 * current_frequency.mean() - current_frequency.var()) / (widths[i] ** 2)

            return costs, widths, counts

        cost_function_values, bucket_widths, bucket_counts = l2_risk_function(array)
        optimal_index = np.argmin(cost_function_values)
        optimal_width = bucket_widths[optimal_index]

        bucket_edges = np.linspace(array.min(), array.max(), bucket_counts[optimal_index] + 1)

        plt.rc('text', usetex=True)

        plt.figure()
        plt.hist(array, range(bucket_edges.size))
        plt.title('Histogram')
        plt.ylabel('Frequency')
        plt.xlabel('Value')
        plt.show()

        plt.figure()
        plt.plot(bucket_widths, cost_function_values, '.b', optimal_width, min(cost_function_values), '*r')
        plt.title('Estimated Loss Function')
        plt.ylabel('Loss')
        plt.xlabel('Bin count')
        plt.show()

        return optimal_width

    if WIDTH_METHOD == 'scott':
        bin_width = scott(samples)
    elif WIDTH_METHOD == 'fd':
        bin_width = freedman_diaconis(samples)
    elif WIDTH_METHOD == 'ss':
        bin_width = shimazaki_shinomoto(samples)
    else:
        raise KeyError('invalid bin width method')

    return bin_width


def calculate_bin_count(samples, bin_width):
    def base_choice(array, width):
        count = np.ceil((array.max() - array.min()) / width)
        count = int(count.item())
        return count

    def sqrt_choice(array):
        count = np.ceil(np.sqrt(len(array)))
        return count

    def sturge(array):
        count = np.ceil(np.log2(len(array))) + 1
        return count

    def rice(array):
        count = np.ceil(2 * np.cbrt(len(array)))
        return count

    def doane(array):
        n = len(array)
        sig_g_1 = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        g_1 = skew(array)
        # g_1 = moment(array, moment=3)
        count = 1 + np.log2(n) + np.log2(1 + (np.abs(g_1) / sig_g_1))
        return count

    if COUNT_METHOD == 'base':
        bin_count = base_choice(samples, bin_width)
    elif COUNT_METHOD == 'sqrt':
        bin_count = sqrt_choice(samples)
    elif COUNT_METHOD == 'sturge':
        bin_count = sturge(samples)
    elif COUNT_METHOD == 'rice':
        bin_count = rice(samples)
    elif COUNT_METHOD == 'doane':
        bin_count = doane(samples)
    else:
        raise KeyError('invalid bin count method')

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
    rewards_original = dict(zip(original_domain.mdp[consts.COL_STATE_TO], original_domain.mdp[consts.COL_REWARD]))

    rewards_aggregate = dict()
    for s_original, s_aggregate in state_mapping.items():
        rewards_aggregate.setdefault(s_aggregate, int())
        rewards_aggregate[s_aggregate] += rewards_original[s_original]

    return rewards_original, rewards_aggregate


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
                                  discount=GAMMA,
                                  threshold=TAU,
                                  verbose=True)
    agent = BaseAgent(domain,
                      solver,
                      epochs=STEPS)
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
                                            discount=GAMMA,
                                            threshold=TAU,
                                            verbose=True)
    agent_aggregate = BaseAgent(domain=domain_aggregate,
                                solver=solver_aggregate,
                                epochs=STEPS)
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
    plt.savefig(consts.RES_PATH + 'steps_rewards.png')
    plt.show()

    plot_cumulative(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.savefig(consts.RES_PATH + 'cumulative_rewards.png')
    plt.show()

    plot_rolling(rewards, rewards_aggregate, rewards_aggregate_adapted)
    plt.savefig(consts.RES_PATH + 'rolling_rewards.png')
    plt.show()


if __name__ == '__main__':
    machine_replacement_domain = MachineReplacementMDPDomain
    river_swim_domain = RiverSwimMDPDomain
    six_arms_domain = SixArmsMDPDomain
    original_rewards, aggregate_rewards, adapted_rewards = run(machine_replacement_domain)
    visualize_rewards(original_rewards, aggregate_rewards, adapted_rewards)
