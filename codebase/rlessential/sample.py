class Sample:

    def __init__(self, current_state, action, reward, next_state, terminal=False):
        self.current_state = current_state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal

    def __repr__(self):
        """Create string representation of tuple."""
        return 'Sample({}, {}, {}, {}, {})\n'.format(self.current_state,
                                                   self.action,
                                                   self.reward,
                                                   self.next_state,
                                                   self.terminal)
