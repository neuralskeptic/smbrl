from src.utils.set_evaluation import SetEval


def replay_agent(agent, core, n_episodes_test, verbose=False, render=False):
    with SetEval(agent):
        data = core.evaluate(
            n_episodes=n_episodes_test, quiet=not verbose, render=render
        )
    return data
