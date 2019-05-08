import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from player import CartPolePlayer

"""
     Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
    Solved Requirements:
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""


if __name__ == "__main__":

    policies = [{'dir': 'initial', 'q': False},
                {'dir': 'qvals', 'q': True}]
    files_name = {'policy': '/policy_nn.csv', 'scale': '/scales.csv', 'q_values': '/qvalues.csv'}

    trials = 1000
    render = False
    epsilon = 1

    for p in policies:
        path_policy = p['dir'] + files_name['policy']
        path_scale = p['dir'] + files_name['scale']
        path_q_values = p['dir'] + files_name['q_values']

        policy = pd.read_csv(path_policy)
        scale = pd.read_csv(path_scale).values
        q_values = pd.read_csv(path_q_values) if p['q'] else None

        cart_pole = CartPolePlayer(policy, scale, q_values, render)
        total_rewards = cart_pole.play(trials, epsilon)

        average_reward = sum(total_rewards.values()) / trials
        print("Average reward per trial", average_reward)

        x_axis = np.array([i for i in range(trials)])
        y_axis = np.array([i for i in total_rewards.values()])

        cm = plt.cm.get_cmap('viridis')
        plt.figure()
        sns.lineplot(x_axis, y_axis)
        sns.set()

        plt.title("Reward per episode | {} | eps={} | avg. reward={}"
                  .format(p['dir'], epsilon, average_reward))
        plt.show()
