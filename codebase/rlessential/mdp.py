import abc


class MDP(object):

    @abc.abstractmethod
    def state_count(self):
        """
        Counts the states in the MDP

        :return: number of states
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def action_count(self):
        """
        Counts the actions in the MDP

        :return: number of actions
        """
        pass  # pragma: no cover


class TwoStateMDP:

    def __init__(self, mdp):
        self.mdp = mdp
        self.num_states = self.state_count()
        self.num_actions = self.action_count()

    def state_count(self):
        unique_from = self.mdp['idstatefrom'].nunique()
        unique_to = self.mdp['idstateto'].nunique()

        return max(unique_from, unique_to)

    def action_count(self):
        unique_action = self.mdp['idaction'].nunique()

        return unique_action
