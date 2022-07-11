import json
import os

import numpy as np
import quanser_robots
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from src.utils.replay_agent import replay_agent
from src.utils.seeds import fix_random_seed

PATH_TO_EXPERIMENT = "./logs/tmp/SAC/Qube-100-v0/0/2022-07-11--18-41-16"
AGENT_EPOCH = "2"

args = json.load(open(os.path.join(PATH_TO_EXPERIMENT, "args.json")))

# MDP
mdp = Gym(args["env_id"], gamma=args["gamma"])

# Fix seed
fix_random_seed(args["seed"], mdp=None)

# Agent
agent_path = os.path.join(PATH_TO_EXPERIMENT, f"agent_{AGENT_EPOCH}.msh")
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
