import json
import os

import numpy as np
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from scripts.on_policy.environments import MAZE_ENVIRONMENTS
from src.utils.eval import replay_agent
from src.utils.seeds import fix_random_seed

# PATH_TO_EXPERIMENT = '/home/carvalho/Projects/MVD/mvd-stepbased/scripts/on_policy/logs/ppo_trpo_2022-03-30_16-49-00/env_id___PointMassMaze00/alg___ppo/0'
PATH_TO_EXPERIMENT = "/home/carvalho/Projects/MVD/mvd-stepbased/scripts/on_policy/logs/tmp/exp_tree_pg/pendulum/0"
AGENT_EPOCH = "19"

args = json.load(open(os.path.join(PATH_TO_EXPERIMENT, "args.json")))

# MDP
env_id = args["env_id"]
if env_id in MAZE_ENVIRONMENTS:
    env_d = MAZE_ENVIRONMENTS[env_id]
    mdp = env_d["env"](gamma=args["gamma"], **env_d["kwargs"])
else:
    mdp = Gym(env_id, gamma=args["gamma"])

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
