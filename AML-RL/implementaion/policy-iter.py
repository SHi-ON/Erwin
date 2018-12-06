import os  # module that handles OS features
import sys  # sys module needed to access command line information
import re
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Policy Iteration')
parser.add_argument('csv_file')
parser.add_argument('-o', default='out')

args = parser.parse_args()

csv_file = args.csv_file
out_file = args.o

idstatefrom = []
idaction = []
idstateto = []
probability = []
reward = []
states = {}
policy_a = []
V = []

GAMMA = 0.75

'''
Read file and properly setup stucture for policy iteration
'''


def read_csv_file():
    global V
    with open(csv_file) as infile:
        i = 0
        for line in infile:
            if i == 0:
                i += 1
                continue
            columns = line.split(',')
            state = int(columns[0])
            action = int(columns[1])
            if state in states:
                if action in states[state]:
                    states[state][action].append(line)
                else:
                    states[state][action] = []
                    states[state][action].append(line)
            else:
                states[state] = {}
                states[state][action] = []
                states[state][action].append(line)
                V.append(0)
                policy_a.append(0)

            line = line.rstrip(os.linesep)
            # print(line)
    infile.close()


'''
Perform policy iteration
'''


def evaluate():
    global V

    i = 0
    print('idstate,idaction,bestV')
    policy_prev = [-1, -1, -1, -1]
    while (policy_prev != policy_a):
        print('Iteration ' + str(i))
        policy_prev = policy_a.copy()
        mV = np.matrix(V)
        mV = mV.transpose()

        for state, state_value in states.items():
            pol_best = 0
            for action, action_value in state_value.items():

                T = []
                R = []

                for a in action_value:
                    columns = a.split(',')
                    T.append(float(columns[3]))
                    R.append(float(columns[4]))

                mT = np.matrix(T)
                mR = np.matrix(R)
                mR = mR.transpose()

                mB = np.add(mR, GAMMA * (mV))

                improved = np.matmul(mT, mB)
                improvedf = float(improved.item(0))

                if improvedf > pol_best:
                    pol_best = improvedf
                    policy_a[state] = action
                    V[state] = improvedf

        j = 0
        while j < len(policy_a):
            print(j, policy_a[j], V[j])
            j += 1
        i += 1
        print('')


def main():
    read_csv_file()
    evaluate()
    print('idstate,idaction')
    i = 0

    out = open(out_file, 'w')
    out.write('idstate,idaction\n')

    while i < len(policy_a):
        print(i, policy_a[i])
        out.write(str(i) + ',' + str(policy_a[i]) + '\n')
        i += 1
    out.close()


if __name__ == '__main__':
    main()
