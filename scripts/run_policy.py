import argparse
import json
import os

import numpy as np
import quanser_robots
import yaml
from experiment_launcher import run_experiment
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from src.utils.replay_agent import replay_agent
from src.utils.seeds import fix_random_seed


def render_policy(
    results_dir: str = "../logs/tmp/SAC/Qube-500-v0/10/2022-07-13--09-13-26",
    agent_epoch: str = "end",
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
    data = replay_agent(agent, core, 5, verbose=False, render=True)
    J = compute_J(data, args["gamma"])
    R = compute_J(data)
    print(f"J: {np.mean(J)}\nR: {np.mean(R)}")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(render_policy)
