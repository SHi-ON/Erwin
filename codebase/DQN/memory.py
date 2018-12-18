import numpy as np


class ExperienceReplay(object):

    def __init__(self, mem_len=100, disc=0.9):
        self.mem_size = mem_len
        self.disc = disc
        self.memory = list()

    def feature_dimension(self):
        return self.memory[0][0][0].shape[1]

    def store(self, record, game_over):
        # memory[i] = [[s, a, r, s'], game_over]
        self.memory.append([record, game_over])
        if len(self.memory) > self.mem_size:
            del self.memory[0]

    # applying Bellman equation (update)
    def get_batch(self, model, batch_size=10):
        memory_len = len(self.memory)
        # number of possible actions
        num_actions = model.output_shape[-1]
        # dimension of the input features
        phi_dim = self.feature_dimension()
        # min(len_memory, batch_size) <=> inputs.shape[0]
        rows = min(memory_len, batch_size)

        # initializing input features.
        # min just in case that still not many states observed
        inputs = np.zeros((rows, phi_dim))

        # initializing target action-values (Q-table)
        # target: Q(s,a) = r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros((rows, num_actions))

        for i, num in enumerate(np.random.randint(0, memory_len, size=rows)):
            # random sampling to break correlations
            # (s, a, r, s')
            s, a, r, s_p = self.memory[num][0]
            game_over = self.memory[num][1]

            # storing state s in the inputs
            inputs[i:i + 1] = s

            # targets Qs
            targets[i] = model.predict(s)[0]
            # max_a' Q(s', a')
            max_q_sp = np.max(model.predict(s_p)[0])
            # if it's the last year
            if game_over:
                targets[i, a] = r
            else:
                # r + gamma * max_a' Q(s', a')
                targets[i, a] = r + self.disc * max_q_sp
        return inputs, targets
