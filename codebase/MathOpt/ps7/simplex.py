# Linear Programming solving by the simplex method
# by Shayan Amani - spring 2019
# https://shayanamani.com

import numpy as np


class Simplex:

    def __init__(self, c, A, b):
        self.c = c.astype(float) * -1
        self.A = A.astype(float)
        self.b = b.astype(float).T

        self.tab = self.init_tableau()

    def get_num_basic_vars(self):
        return self.A.shape[0]

    def get_rhs(self):
        try:
            rhs = self.tab[:, -1]
        except AttributeError:
            rhs = self.b
        return rhs

    def init_tableau(self):
        n_basics = self.get_num_basic_vars()
        slack_vars = np.eye(n_basics)
        rhs = self.get_rhs()

        # note the tuple inside!
        tab_var = np.concatenate((self.A, slack_vars, rhs), axis=1)

        width = tab_var.shape[1] - self.c.shape[1]
        # adds padding
        # pad width: ((top, bottom), (left, right))
        z = np.pad(self.c, ((0, 0), (0, width)), mode='constant')

        return np.concatenate((tab_var, z), axis=0)

    def get_pivot_column(self):
        # pivot column index
        pivot_col_i = np.argmin(self.tab[-1])
        return self.tab[:, pivot_col_i]

    # pivot value and pivot row index
    # using min test
    def get_pivot(self):
        rhs = self.get_rhs()
        pc = self.get_pivot_column()
        pc_filtered = np.where(pc > 0, pc, np.NaN)
        div = rhs / pc_filtered

        pivot_row_index = int(np.where(div == np.nanmin(div))[0][0])
        pivot = pc[pivot_row_index]
        return pivot, pivot_row_index

    # TODO: pretty printing
    def optimal(self):
        return self.tab[:, -1]

    def simplex_solve(self):
        finished = False
        p_val, pr = self.get_pivot()
        pc = self.get_pivot_column()
        tab = self.tab

        # make the pivot value to be 1
        tab[pr, :] = tab[pr, :] / p_val

        # make other pivot entries to be 0
        n = self.get_num_basic_vars()
        cols = tab.shape[1]
        condition = np.array([True] * tab.shape[0])
        condition[pr] = False
        tab[condition, :] = ((-1 * pc[condition].reshape((n, 1))) @ tab[pr].reshape((1, cols))) + tab[condition, :]

        self.tab = tab

        if not np.any(self.tab[-1, :] < 0):
            finished = True

        return tab, finished
