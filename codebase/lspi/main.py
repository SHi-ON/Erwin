import warnings

import numpy as np
import pandas as pd

from domains import ChainWalkDomain, RiverSwimDomain, SixArmsDomain
from basis_functions import FakeBasis, OneDimensionalPolynomialBasis, RadialBasisFunction
from policy import Policy
from solvers import LSTDQSolver
from lspi import learn

warnings.simplefilter(action='ignore', category=FutureWarning)


def chain_walk(n_samples):
    domain = ChainWalkDomain(num_states=4, reward_location=ChainWalkDomain.RewardLocation.Middle)

    samples = []
    init_action = np.random.randint(domain.num_actions)
    init_sample = domain.apply_action(init_action)
    samples.append(init_sample)

    for i in range(1, n_samples):
        a = samples[-1].action
        samples.append(domain.apply_action(a))

    # basis = FakeBasis(2)
    poly_basis = OneDimensionalPolynomialBasis(3, 2)
    # policy = Policy(basis)
    policy = Policy(poly_basis)
    policy.weights
    print('initial policy weights:', policy.weights)

    solver = LSTDQSolver()

    learned_policy = learn(samples, policy, solver)
    print('final policy weights:', learned_policy.weights)

    return learned_policy


def mdps(domain, n_samples):
    samples = []
    init_action = np.random.randint(domain.num_actions)
    init_sample = domain.apply_action(init_action)
    samples.append(init_sample)

    for i in range(1, n_samples):
        a = samples[-1].action
        samples.append(domain.apply_action(a))

    # basis = FakeBasis(2)
    # basis = OneDimensionalPolynomialBasis(3, domain.num_actions)
    basis = RadialBasisFunction(np.array([np.array([i]) for i in range(4)]), 0.8, domain.num_actions)
    policy = Policy(basis)
    print('initial policy weights:', policy.weights)

    solver = LSTDQSolver()

    learned_policy = learn(samples, policy, solver)
    print('final policy weights:', learned_policy.weights)

    return learned_policy


def main():
    sample_size = 100
    # max_iteration = 10

    print(ChainWalkDomain.__name__)
    cw_policy = chain_walk(sample_size)

    rs_mdp = pd.read_csv("codebase/adrel/craam/riverswim_mdp.csv")
    rs_domain = RiverSwimDomain(rs_mdp)
    print(RiverSwimDomain.__name__)
    rs_policy = mdps(rs_domain, sample_size)

    sa_mdp = pd.read_csv("codebase/adrel/craam/sixarms_mdp.csv")
    sa_domain = SixArmsDomain(sa_mdp)
    print(SixArmsDomain.__name__)
    sa_policy = mdps(sa_domain, sample_size)


if __name__ == "__main__":
    main()
