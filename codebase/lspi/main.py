import numpy as np
import pandas as pd

from domains import ChainWalkDomain, RiverSwimDomain
from basis_functions import FakeBasis, OneDimensionalPolynomialBasis
from policy import Policy
from solvers import LSTDQSolver
from lspi import learn


def chain_walk(n_samples):
    domain = ChainWalkDomain(num_states=4, reward_location=ChainWalkDomain.RewardLocation.Middle)

    samples = []
    init_action = np.random.randint(domain.num_actions)
    init_sample = domain.apply_action(init_action)
    samples.append(init_sample)

    for i in range(1, n_samples):
        a = samples[-1].action
        samples.append(domain.apply_action(a))

    basis = FakeBasis(2)
    # poly_basis = OneDimensionalPolynomialBasis(3, 2)
    # poly_basis.evaluate(np.array([2]), 1)
    policy = Policy(basis)
    print('initial policy weights:', policy.weights)

    solver = LSTDQSolver()

    learned_policy = learn(samples, policy, solver)
    print('final policy weights:', learned_policy.weights)

    return learned_policy


def river_swim(n_samples):
    mdp = pd.read_csv("codebase/adrel/craam/riverswim_mdp.csv")
    domain = RiverSwimDomain(mdp)

    samples = []
    init_action = np.random.randint(domain.num_actions)
    init_sample = domain.apply_action(init_action)
    samples.append(init_sample)

    for i in range(1, n_samples):
        a = samples[-1].action
        samples.append(domain.apply_action(a))

    basis = FakeBasis(2)
    # basis = OneDimensionalPolynomialBasis(3, 2)
    policy = Policy(basis)
    print('initial policy weights:', policy.weights)

    solver = LSTDQSolver()

    learned_policy = learn(samples, policy, solver)
    print('final policy weights:', learned_policy.weights)

    return learned_policy


def main():
    sample_size = 100
    # max_iteration = 10
    cw_policy = chain_walk(sample_size)
    rs_policy = river_swim(sample_size)


if __name__ == "__main__":
    main()
