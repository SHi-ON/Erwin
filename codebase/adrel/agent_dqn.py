import numpy as np
import random
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class AgentDQN:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # arbitrary size

        self.gamma = 0.95  # reward discount rate

        # exploration rates
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.alpha = 0.001  # learning rate

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # first argument of Dense is dimension of output.
        # last layer's output dimension makes more sense
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # exploration - exploitation
    # returns a random exploratory action or
    # a predicted action from the network
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # TODO: try to see the output shape
        predicted_action_values = self.model.predict(state)[0]
        return np.argmax(predicted_action_values)  # returns action

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                next_action_values = self.model.predict(next_state)[0]
                # q_target = r + gamma * q_s'
                target = (reward + self.gamma * np.amax(next_action_values))
            # updates q network
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # learn the experience
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay exploration and towards exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # TODO: change to a better arg name
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
