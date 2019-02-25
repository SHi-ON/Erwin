import time as _time

import numpy as np
import scipy.sparse as sp

def dimCheck(transition):
    A = len(transition)
    try:
        if transition.ndim == 3:
            S = transition.shape[1]
        else:
            S = transition[0].shape[0]
    except AttributeError:
        S = transition[0].shape[0]
    return S, A


def littleSnitch(iteration, variation):
    # if isinstance(variation, float):
    #     print("{:>10}{:>12f}".format(iteration, variation))
    # elif isinstance(variation, int):
    #     print("{:>10}{:>12d}".format(iteration, variation))
    # else:
        print("{:>8}\t{:>8}".format(iteration, variation))


class ModPolIter(object):

    def __init__(self, transitions, reward, gamma, epsilon,
                 max_iter=10):

        self.gamma = float(gamma)
        self.max_iter = int(max_iter)

        self.S, self.A = dimCheck(transitions)
        self.P = self.transProbCalc(transitions)
        self.R = self.rewardCalc(reward, transitions)

        # the verbosity is by default turned off
        self.verbose = True
        # Initially the time taken to perform the computations is set to None
        self.time = None
        # set the initial iteration count to zero
        self.iter = 0
        # V should be stored as a vector ie shape of (S,) or (1, S)
        self.V = None
        # policy can also be stored as a vector
        self.policy = None

        null = np.zeros(self.S)
        self.policy, null = self._bellmanOperator(null)
        del null

        self.V = np.zeros(self.S)
        self.eval_type = "matrix"
        self.epsilon = float(epsilon)
        assert epsilon > 0, "'epsilon' must be greater than 0."

        # computation of threshold of variation for V for an epsilon-optimal
        # policy
        if self.gamma != 1:
            self.thresh = self.epsilon * (1 - self.gamma) / self.gamma
        else:
            self.thresh = self.epsilon

        if self.gamma == 1:
            self.V = np.zeros(self.S)
        else:
            Rmin = min(R.min() for R in self.R)
            self.V = 1 / (1 - self.gamma) * Rmin * np.ones((self.S,))

        ### Run the modified policy iteration algorithm.

        # self._startRun()
        littleSnitch('Iteration', 'Variation')
        self.time = _time.time()

        while True:
            self.iter += 1

            self.policy, Vnext = self._bellmanOperator()
            # [Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, PR, policy);

            variation = (Vnext - self.V).max() - (Vnext - self.V).min()
            if self.verbose:
                littleSnitch(self.iter, variation)

            self.V = Vnext
            if variation < self.thresh:
                break
            else:
                is_verbose = False
                if self.verbose:
                    self.verbose = False
                    is_verbose = True

                self._evalPolicyIterative(self.V, self.epsilon, self.max_iter)

                if is_verbose:
                    self.verbose = True

        self.V = tuple(self.V.tolist())

        try:
            self.policy = tuple(self.policy.tolist())
        except AttributeError:
            self.policy = tuple(self.policy)

        self.time = _time.time() - self.time

    ##########################Policy Iter

    def _computePpolicyPRpolicy(self):
        Ppolicy = np.empty((self.S, self.S))
        Rpolicy = np.zeros(self.S)
        for aa in range(self.A):  # avoid looping over S
            # the rows that use action a.
            ind = (self.policy == aa).nonzero()[0]
            # if no rows use action a, then no need to assign this
            if ind.size > 0:
                try:
                    Ppolicy[ind, :] = self.P[aa][ind, :]
                except ValueError:
                    Ppolicy[ind, :] = self.P[aa][ind, :].todense()
                Rpolicy[ind] = self.R[aa][ind]
        if type(self.R) is sp.csr_matrix:
            Rpolicy = sp.csr_matrix(Rpolicy)
        return (Ppolicy, Rpolicy)

    def _evalPolicyIterative(self, V0=0, epsilon=0.0001, max_iter=10000):
        try:
            assert V0.shape in ((self.S,), (self.S, 1), (1, self.S)), \
                "'V0' must be a vector of length S."
            policy_V = np.array(V0).reshape(self.S)
        except AttributeError:
            if V0 == 0:
                policy_V = np.zeros(self.S)
            else:
                policy_V = np.array(V0).reshape(self.S)

        policy_P, policy_R = self._computePpolicyPRpolicy()

        if self.verbose:
            littleSnitch("Iteration", "V variation")

        itr = 0
        done = False
        while not done:
            itr += 1

            Vprev = policy_V
            policy_V = policy_R + self.gamma * policy_P.dot(Vprev)

            variation = np.absolute(policy_V - Vprev).max()
            if self.verbose:
                littleSnitch(itr, variation)

            # ensure |Vn - Vpolicy| < epsilon
            if variation < ((1 - self.gamma) / self.gamma) * epsilon:
                done = True
                if self.verbose:
                    print("Iterating stopped, epsilon-optimal value function found.")
            elif itr == max_iter:
                done = True
                if self.verbose:
                    print("Iterating stopped due to maximum number of iterations condition.")

        self.V = policy_V

    def _evalPolicyMatrix(self):

        Ppolicy, Rpolicy = self._computePpolicyPRpolicy()
        # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)* PR
        self.V = np.linalg.solve(
            (sp.eye(self.S, self.S) - self.gamma * Ppolicy), Rpolicy)

    ###################MDP

    def _bellmanOperator(self, V=None):
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (1, self.S)), "V is not the " \
                                                            "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            Q[aa] = self.R[aa] + self.gamma * self.P[aa].dot(V)
        return (Q.argmax(axis=0), Q.max(axis=0))

    def transProbCalc(self, transition):
        return tuple(transition[a] for a in range(self.A))

    def rewardCalc(self, reward, transition):
        try:
            if reward.ndim == 1:
                return self._computeVectorReward(reward)
            elif reward.ndim == 2:
                return self._computeArrayReward(reward)
            else:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
        except (AttributeError, ValueError):
            if len(reward) == self.A:
                r = tuple(map(self._computeMatrixReward, reward, transition))
                return r
            else:
                return self._computeVectorReward(reward)

    def _computeVectorReward(self, reward):
        if sp.issparse(reward):
            raise NotImplementedError
        else:
            r = np.array(reward).reshape(self.S)
            return tuple(r for a in range(self.A))

    def _computeArrayReward(self, reward):
        if sp.issparse(reward):
            raise NotImplementedError
        else:
            def func(x):
                return np.array(x).reshape(self.S)

            return tuple(func(reward[:, a]) for a in range(self.A))

    def _computeMatrixReward(self, reward, transition):
        if sp.issparse(reward):
            return reward.multiply(transition).sum(1).A.reshape(self.S)
        elif sp.issparse(transition):
            return transition.multiply(reward).sum(1).A.reshape(self.S)
        else:
            return np.multiply(transition, reward).sum(1).reshape(self.S)

#####################################Mod Pol Iter




if __name__ == "__main__":
    trans_probs = np.array([[[0.3, 0.2, 0.1, 0.4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]])
    rewards = np.array([[[1.0, 1.5, 1.8, 2.0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    mpi = ModPolIter(trans_probs, rewards, 0.95, 0.01, 100000000)

    print(mpi.policy)
    print(mpi.V)

