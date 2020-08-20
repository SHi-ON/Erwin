import numpy as np
from sklearn.cluster import KMeans

import consts
import methods
from agents import BaseAgent
from domains import MachineReplacementMDPDomain, RiverSwimMDPDomain, SixArmsMDPDomain
from solvers import ValueIterationSolver
from visualization.plot import visualize_rewards

# hyper-parameters
GAMMA = 0.90
EPSILON = 0.0001
TAU = (EPSILON * (1 - GAMMA)) / (2 * GAMMA)

# ['sqrt', 'sturge', 'rice', 'doane', 'scott', 'freedman-diaconis', 'shimazaki-shinomoto']
ALGORITHMS = [{'method': 'sqrt', 'steps': 90},
              {'method': 'sturge', 'steps': 500},
              {'method': 'rice', 'steps': 120},
              {'method': 'doane', 'steps': 125},

              {'method': 'scott', 'steps': 2600},
              {'method': 'freedman-diaconis', 'steps': 3750},
              {'method': 'shimazaki-shinomoto', 'steps': 10000}]

i = 6
METHOD = ALGORITHMS[i]['method']
STEPS = ALGORITHMS[i]['steps']


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


7


def select_bin_counts(samples):
    """
    Calculates bin-counts via the given method.

    :param samples: collected samples
    :type samples: np.ndarray

    :return: bin-count
    :rtype: int
    """
    if METHOD == 'sqrt':
        bin_count = methods.sqrt_choice(samples)
    elif METHOD == 'sturge':
        bin_count = methods.sturge(samples)
    elif METHOD == 'rice':
        bin_count = methods.rice(samples)
    elif METHOD == 'doane':
        bin_count = methods.doane(samples)
    elif METHOD == 'scott':
        bin_width = methods.scott(samples)
        bin_count = methods.base_count(samples, bin_width)
    elif METHOD == 'freedman-diaconis':
        bin_width = methods.freedman_diaconis(samples)
        bin_count = methods.base_count(samples, bin_width)
    elif METHOD == 'shimazaki-shinomoto':
        bin_width = methods.shimazaki_shinomoto(samples)
        bin_count = methods.base_count(samples, bin_width)
    else:
        raise KeyError('invalid bin-count selection method')

    bin_count = int(bin_count)
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

    bucket_count = select_bin_counts(samples=states)
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
    print('bin count:', bucket_count)

    return rewards, rewards_aggregate, rewards_aggregate_adapted


if __name__ == '__main__':
    machine_replacement_domain = MachineReplacementMDPDomain
    river_swim_domain = RiverSwimMDPDomain
    six_arms_domain = SixArmsMDPDomain
    original_rewards, aggregate_rewards, adapted_rewards = run(machine_replacement_domain)
    visualize_rewards(original_rewards, aggregate_rewards, adapted_rewards, METHOD)
