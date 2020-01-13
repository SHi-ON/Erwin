import numpy as np

from agents import QLearningCartPoleAgent, QLearningMountainCarAgent, CartPoleAgent


def q_learning():
    # discounted-reward problem
    agent = QLearningCartPoleAgent(num_buckets=(1, 2, 6, 12), num_episodes=1000, discount=0.99)
    # agent = QLearningMountainCarAgent(num_buckets=(6, 6), num_episodes=10000, discount=0.99)
    agent.train()
    agent.run()




if __name__ == '__main__':
    q_learning()
    # agent = CartPoleAgent()
    # agent.run()
    # samples = agent.samples
    # samples = np.array(samples)




