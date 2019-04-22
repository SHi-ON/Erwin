# Linear Programming
# The simplex method and Gurobi solver comparison
# by Shayan Amani - spring 2019
# https://shayanamani.com

# The LP is assumed to be in the standard form
import numpy as np
from gurobipy import *
from simplex import Simplex

# set precision for printing ndarrays
np.set_printoptions(precision=4, formatter={'float': lambda x: str(round(x, 4))})


def simplex_optimize():
    c = np.array([[6, 5, 4]])
    A = np.array([[2, 1, 1],
                  [1, 3, 2],
                  [2, 1, 2]])
    b = np.array([[240, 360, 300]])

    simplex = Simplex(c, A, b)
    optimal = False
    count = 0
    while not optimal:
        tab, optimal = simplex.simplex_solve()
        count += 1
        print('-------Tableau {}-------'.format(count))
        print(tab.astype('float'))
    optimal = simplex.optimal()

    print('x1* = {} \n x2* = {} \n x3* = {}'.format(optimal[0], optimal[1], 0))
    print('Optimal objective = {}'.format(optimal[-1]))


def gurobi_optimize():
    try:

        # Create a new model
        m = Model("mip1")

        # Create variables
        # variable type is CONTINUOUS by default
        x1 = m.addVar(name="x1")
        x2 = m.addVar(name="x2")
        x3 = m.addVar(name="x3")

        # Set objective
        m.setObjective(6 * x1 + 5 * x2 + 4 * x3, GRB.MAXIMIZE)

        # Add constraint
        m.addConstr(2 * x1 + 1 * x2 + 1 * x3 <= 240, "c0")
        m.addConstr(1 * x1 + 3 * x2 + 2 * x3 <= 360, "c1")
        m.addConstr(2 * x1 + 1 * x2 + 2 * x3 <= 300, "c2")

        m.optimize()

        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))

        print('Obj: %g' % m.objVal)

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


if __name__ == "__main__":
    print('=======Simplex Method=======')
    simplex_optimize()

    print('=======Gurobi Optimization=======')
    gurobi_optimize()
