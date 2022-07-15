import argparse
import json
import os

import numpy as np
import quanser_robots
import yaml
from experiment_launcher import run_experiment
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from src.utils.replay_agent import replay_agent
from src.utils.seeds import fix_random_seed


def render_policy(
    results_dir: str = "../logs/good/SAC/Qube-100-v0/9/2022-07-12--20-45-13",
    agent_epoch: str = "end",
    render: bool = False,
    plot: bool = True,
    seed: int = -1,  ## IGNORED (only needed to run with with run_experiment)
):
    try:
        with open(os.path.join(results_dir, "args.json")) as f:
            args = json.load(f)
    except:
        with open(os.path.join(results_dir, "args.yaml")) as f:
            args = yaml.load(f, Loader=yaml.Loader)

    # MDP
    mdp = Gym(args["env_id"], gamma=args["gamma"])

    # Fix seed
    fix_random_seed(args["seed"], mdp=None)

    # Agent
    agent_path = os.path.join(results_dir, f"agent_{agent_epoch}.msh")
    agent = Agent.load(agent_path)
    prepro = None
    if args["preprocess_states"]:
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)
    core = Core(agent, mdp, preprocessors=[prepro] if prepro is not None else None)

    # Evaluate
    # data = replay_agent(agent, core, 5, verbose=False, render=render)
    # J = compute_J(data, args["gamma"])
    # R = compute_J(data)
    # print(f"mean J: {np.mean(J)}\n mean R: {np.mean(R)}")

    # Plot
    if plot:
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        MAX_STEPS = 100 + 80
        for i in range(5):
            data = replay_agent(agent, core, 1, verbose=False, render=render)
            s, a, r, ss, absorb, last = parse_dataset(data)
            axs[0].plot(r[:MAX_STEPS])
            axs[1].plot(a[:MAX_STEPS])
        axs[0].set_title("run policy")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[1].set_xlabel("steps")
        axs[0].set_ylabel("reward")
        axs[1].set_ylabel("action")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(render_policy)
