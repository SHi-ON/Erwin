from agents import QLearningCartPoleAgent

if __name__ == '__main__':
    # discounted-reward problem
    agent = QLearningCartPoleAgent(num_buckets=(1000, 1000, 6000, 12000), num_episodes=1000, discount=0.99)
    agent.train()
    agent.run()
