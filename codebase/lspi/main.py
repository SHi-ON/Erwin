import numpy as np
import pandas as pd

from domains import ChainWalkDomain, RiverSwimDomain
from basis_functions import FakeBasis, OneDimensionalPolynomialBasis
from policy import Policy
from solvers import LSTDQSolver
from lspi import learn

# os.chdir("/home/sa1149/PycharmProjects/Erwin/codebase/LSPI")

##### ChainWalk
samples = []

domain = ChainWalkDomain(num_states=4, reward_location=ChainWalkDomain.RewardLocation.Middle)
domain.reset()

init_action = np.random.randint(domain.num_actions())
init_sample = domain.apply_action(init_action)
samples.append(init_sample)

for i in range(1, 101):
    a = samples[-1].action
    samples.append(domain.apply_action(a))

basis = FakeBasis(2)
poly_basis = OneDimensionalPolynomialBasis(3, 2)
poly_basis.evaluate(np.array([2]), 1)

policy = Policy(basis)
print('initial policy weights:', policy.weights)

solver = LSTDQSolver()

ret = learn(samples, policy, solver, max_iterations=100)
print('final policy weights:', ret.weights)
policy.weights

if False: del policy, ret


##### RiverSwim
rs_mdp = pd.read_csv("codebase/adrel/craam/riverswim_mdp.csv")

samples = []

domain = RiverSwimDomain(rs_mdp)
domain.reset()

init_action = np.random.randint(domain.num_actions)
init_sample = domain.apply_action(init_action)
samples.append(init_sample)

for i in range(1, 101):
    a = samples[-1].action
    samples.append(domain.apply_action(a))

# basis = OneDimensionalPolynomialBasis(3, 2)
basis = FakeBasis(2)
policy = Policy(basis)
print('initial policy weights:', policy.weights)

solver = LSTDQSolver()

ret = learn(samples, policy, solver, max_iterations=100)
print('final policy weights:', ret.weights)

if False: del policy, ret
