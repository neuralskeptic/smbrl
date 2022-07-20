import json
import os

import pandas as pd
import yaml
from experiment_launcher import run_experiment
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from src.utils.replay_agent import replay_agent
from src.utils.seeds import fix_random_seed


def render_policy(
    results_dir: str = "../models/2022_07_15__14_57_42",
    agent_epoch: str = "end",
    render: bool = False,
    plot: bool = True,
    export: bool = False,
    n_steps_export: int = 25e3,
    seed: int = -1,  ## IGNORED (only needed to run with with run_experiment)
):
    try:
        with open(os.path.join(results_dir, "args.json")) as f:
            args = json.load(f)
    except:
        with open(os.path.join(results_dir, "args.yaml")) as f:
            args = yaml.load(f, Loader=yaml.Loader)

    # MDP
    mdp = Gym(args["env_id"], horizon=args["horizon"], gamma=args["gamma"])

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

    # MAX_STEPS = 100 + 80
    for i in range(5):
        data = replay_agent(agent, core, 1, verbose=False, render=render)
        if plot:
            s, a, r, ss, absorb, last = parse_dataset(data)
            # axs[0].plot(r[:MAX_STEPS])
            # axs[1].plot(a[:MAX_STEPS])
            axs[0].plot(r)
            axs[1].plot(a)
    if plot:
        axs[0].set_title("run policy")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[1].set_xlabel("steps")
        axs[0].set_ylabel("reward")
        axs[1].set_ylabel("action")

    # export data for behavioural cloning
    if export:
        export_data = core.evaluate(n_steps=n_steps_export, render=False, quiet=True)
        s, a, r, ss, absorb, last = parse_dataset(export_data)
        N = len(a)
        df = pd.DataFrame()
        df[["s0", "s1", "s2", "s3", "s4", "s5"]] = s.reshape(N, -1)
        df[["a"]] = a.reshape(N, -1)
        df[["r"]] = r.reshape(N, -1)
        df[["ss0", "ss1", "ss2", "ss3", "ss4", "ss5"]] = ss.reshape(N, -1)
        df[["absorb"]] = absorb.reshape(N, -1)
        df[["last"]] = last.reshape(N, -1)
        datafile = os.path.join(results_dir, f'SAC_on_{args["env_id"]}_{N}.pkl.gz')
        df.to_pickle(datafile)


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(render_policy)
