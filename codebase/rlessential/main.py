import numpy as np

from agents import QLearningCartPoleAgent, QLearningMountainCarAgent, CartPoleAgent, RAAMAgent


def q_learning():
    # discounted-reward problem
    agent = QLearningCartPoleAgent(num_buckets=(1, 2, 6, 12), num_episodes=1000, discount=0.99)
    # agent = QLearningMountainCarAgent(num_buckets=(6, 6), num_episodes=10000, discount=0.99)
    agent.train()
    # agent.run(False)
    agent.simulate()
    return agent.batch

def robust_learning():
    agent = CartPoleAgent()
    agent.run()
    samples = agent.samples
    samples = np.array(samples)
    print(samples.shape)
    env = agent.env

    raam = RAAMAgent(samples=samples, env=env)
    raam.aggregate_states()
    len(raam.S)
    raam.A.shape
    raam.B.shape


if __name__ == '__main__':
    samples = q_learning()
    sample_size = len(samples)
    print('Sample size:', sample_size)
    # step per episode
    avg_spe = (sample_size / 1000)
    print('Average step per episode:', avg_spe)


    # robust_learning()





