class AgentRandom:
    """
    A simple random agent
    """

    def __init__(self, env):
        self.env = env
    def act(self, obs):
        return self.env.action_space.sample()
    def store(self, obs, action, new_obs, reward):
        pass

class AgentPolicy:
    """
    Agent following a policy pi : pi is a dictionary state -> action
    """

    def __init__(self, env, pi):
        self.env = env
        self.pi = pi
    def act(self, obs):
        return self.pi[obs]
    def store(self, obs, action, new_obs, reward):
        pass