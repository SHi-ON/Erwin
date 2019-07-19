import numpy as np
# random package is thread-safe and faster than NumPy.
import random
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class AgentDQN:

    def __init__(self, space_state, space_action):
        self.space_state = space_state
        self.space_action = space_action

        # arbitrary size
        self.memory = deque(maxlen=2000)

        # negative reward
        self.penalty = -10

        # learning rate
        self.alpha = 0.001

        # reward discount rate
        self.gamma = 0.95

        # exploration rates
        # from exploration to exploitation
        self.epsilon = 0.6
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001

        self.model = self._build_model()

    def _build_model(self):
        """
        Configures the neural network's layers and parameters.
        """
        model = Sequential()
        # first argument of Dense is dimension of output,
        # that last layer's output dimension makes sense in this case.
        model.add(Dense(24, input_dim=self.space_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.space_action, activation='linear'))
        # the loss function and optimizer selection
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience (observation) <S, A, R, S'> and status flag, done,
        in form of a tuple.

        :param state: current state
        :param action: taken action
        :param reward: observed reward
        :param next_state: next state
        :param done: end of episode flag
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Action selection based on the exploration-exploitation trade-off.
        Chooses an exploratory action or a predicted one from the neural network.

        The predicted action will be the action with highest Q-value.

        :param state: current state
        :return: selected action
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.space_action)
        predicted_action_values = self.model.predict(state)[0]
        return np.argmax(predicted_action_values)

    def replay(self, batch_size):
        """
        Samples from the past experiences,
        as in Experience Replay in the DQN algorithm.

        Samples once the memory length is greater than batch_size.

        :param batch_size: size of mini-batch
        """
        if len(self.memory) <= batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                next_action_values = self.model.predict(next_state)[0]
                # q_target = r + gamma * q_s'
                target = (reward + self.gamma * np.amax(next_action_values))
            target_f = self.model.predict(state)
            # updates q network
            target_f[0][action] = target
            # learn the experience
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decrease exploration in favor of exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
