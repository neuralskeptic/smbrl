"""
use to switch between evaluation and exploration policies, e.g.:

class MyAgent():
    def set_eval(self, flag):
    if flag:
        # For evaluation, switches the agent's policy to the optimization policy
        self.policy = self.optimization_policy
    else:
        self.policy = self.exploration_policy

agent = MyAgent()

with SetEval(agent):
    data = agent.evaluate(n_episodes=n_episodes_test, quiet=not verbose, render=False)
"""


class SetEval:
    def __init__(self, agent):
        self._agent = agent

    def __enter__(self):
        if hasattr(self._agent, "set_eval"):
            self._agent.set_eval(True)
        else:
            # raise NotImplementedError
            pass
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self._agent, "set_eval"):
            self._agent.set_eval(False)
        else:
            # raise NotImplementedError
            pass
