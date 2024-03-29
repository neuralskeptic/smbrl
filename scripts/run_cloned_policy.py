import json
import os

import gpytorch
import pandas as pd
import quanser_robots
import torch
import yaml
from experiment_launcher import run_experiment
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import parse_dataset
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor
from sklearn.model_selection import train_test_split
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from src.models.gp_models import ExactGPModel
from src.models.linear_bayesian_models import (
    NeuralLinearModel,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.utils.conversion_utils import df2torch, np2torch, qube_rollout2df
from src.utils.replay_agent import replay_agent
from src.utils.seeds import fix_random_seed


def render_policy(
    # results_dir: str = "logs/good/gp_action/3/2022_09_14__14_38_25",
    # results_dir: str = "logs/tmp/nlm_clone_SAC/0/2022_10_19__22_44_56",  # good 5e-4
    results_dir: str = "logs/tmp/nlm_clone_SAC/0/2022_10_20__21_09_00",  # good 1e-4
    # results_dir: str = "logs/tmp/snngp_clone_SAC/0/2022_10_20__18_18_11",  # good 1e-4
    # results_dir: str = "logs/tmp/snngp_clone_SAC/0/2022_10_20__20_38_07",  # good 5e-4
    agent_epoch: str = "end",
    use_cuda: bool = True,  # gp too slow on cpu
    stoch_preds: bool = False,  # sample from pred post; else use mean
    fast_gp: bool = True,  # with gpytorch.settings.fast_pred_var()
    n_runs: int = 10,
    # render: bool = True,
    render: bool = False,
    # plot: bool = False,
    plot: bool = True,
    show_plots: bool = False,
    seed: int = 0,
):
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
    try:
        with open(os.path.join(repo_dir, results_dir, "args.json")) as f:
            args = json.load(f)
    except:
        with open(os.path.join(repo_dir, results_dir, "args.yaml")) as f:
            args = yaml.load(f, Loader=yaml.Loader)

    alg = args["alg"]
    assert repo_dir == args["repo_dir"]
    dataset_file = args["dataset_file"]
    lr = args["lr"]
    # model_save_frequency = args["model_save_frequency"]
    # n_epochs = args["n_epochs"]
    n_trajectories = args["n_trajectories"]
    # wandb_group = args["wandb_group"]

    # load mdp config from cloned policy
    sac_results_dir = os.path.dirname(os.path.join(repo_dir, dataset_file))
    try:
        with open(os.path.join(sac_results_dir, "args.json")) as f:
            sac_args = json.load(f)
    except:
        with open(os.path.join(sac_results_dir, "args.yaml")) as f:
            sac_args = yaml.load(f, Loader=yaml.Loader)

    # MDP
    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])

    # Fix seed
    fix_random_seed(args["seed"])

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # load data (trajectories of the cloned policy)
    # (gp constructor needs the full data; nlm needs data sizes)
    # df: [s0-s5, a, r, ss0-ss5, absorb, last]
    df = pd.read_pickle(os.path.join(repo_dir, dataset_file))
    df = df[df["traj_id"] < n_trajectories]  # take only n_trajectories
    df = df.astype("float32")  # otherwise spectral_norm complains
    traj_dfs = [
        traj.reset_index(drop=True) for (traj_id, traj) in df.groupby("traj_id")
    ]
    train_traj_dfs, test_traj_dfs = train_test_split(traj_dfs, test_size=0.2)
    x_cols = ["s0", "s1", "s2", "s3", "s4", "s5"]
    y_cols = ["a"]
    train_df = pd.concat(train_traj_dfs)
    test_df = pd.concat(test_traj_dfs)
    train_x_df, train_y_df = train_df[x_cols], train_df[y_cols]
    test_x_df, test_y_df = test_df[x_cols], test_df[y_cols]
    x_train = df2torch(train_x_df).to(device)
    y_train = df2torch(train_y_df).to(device)
    x_test = df2torch(test_x_df).to(device)
    # y_test = df2torch(test_y_df).to(device)
    # reshape labels, because mean_prior implicitly flattens and it crashes
    y_train = y_train.reshape(-1)
    # y_test = y_test.reshape(-1)

    # Agent (3 agents and 3 pred_fn's for gp, nlm, snngp)
    agent_path = os.path.join(repo_dir, results_dir, f"agent_{agent_epoch}.pth")
    if alg == "gp":
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=None).to(
            device
        )
        agent = ExactGPModel(x_train, y_train, likelihood).to(device)
        state_dict = torch.load(agent_path)
        agent.load_state_dict(state_dict)

        def pred_fn(state):
            def _pred():
                post_pred = likelihood(agent(state))
                if stoch_preds:  # sample 1
                    action = post_pred.sample()
                else:  # use mean
                    action = post_pred.mean
                # # clear gpu memory (necessary when nograd?)
                # del post_pred
                # torch.cuda.empty_cache()
                return action

            if fast_gp:
                with gpytorch.settings.fast_pred_var():
                    return _pred()
            else:
                return _pred()

        # no training
        agent.eval()
        likelihood.eval()
    elif alg == "nlm" or alg == "snngp":  # shared code, except model
        dim_in = len(x_cols)
        dim_out = len(y_cols)
        if alg == "nlm":
            model = NeuralLinearModel
        else:  # snngp
            model = SpectralNormalizedNeuralGaussianProcess

        agent = model(dim_in, dim_out, args["n_features"])
        agent.to(device)
        state_dict = torch.load(agent_path)
        agent.load_state_dict(state_dict)

        def pred_fn(state):
            mu, covariance, covariance_feat, covariance_out = agent(state)
            if stoch_preds:  # sample 1
                mvn = MultivariateNormal(loc=mu, covariance_matrix=covariance)
                action = mvn.rsample()
            else:  # use mean
                action = mu
            return action

    else:
        raise Exception(f"Agent algorithm -{alg}- unknown. Aborting")

    # Plot
    if plot:
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))

    for i in range(n_runs):
        ##### RUN AGENT for one episode #####
        dataset = list()
        episode_steps = 0
        # steps_progress_bar = tqdm(disable=True)
        # episodes_progress_bar = tqdm(total=self._n_episodes,
        #                              dynamic_ncols=True, disable=quiet,
        #                              leave=False)
        # reset
        mdp.seed(seed + i)
        state = mdp.reset(None).copy()

        last = False
        while not last:
            with torch.no_grad():
                state_torch = np2torch(state).reshape([-1, len(x_cols)]).to(device)
                action_torch = pred_fn(state_torch)
                action = action_torch.to("cpu").reshape(-1).numpy()
            next_state, reward, absorbing, _ = mdp.step(action)
            if render:
                mdp.render()
            episode_steps += 1
            if alg == "gp":
                print(episode_steps)  # because the gp is soooo slow
            # steps_progress_bar.update(1)
            if episode_steps >= mdp.info.horizon or absorbing:
                last = True
                # episodes_progress_bar.update(1)
            next_state = next_state.copy()  # why this?
            sample = (state, action, reward, next_state, absorbing, last)
            dataset.append(sample)

            state = next_state

        mdp.stop()
        # steps_progress_bar.close()
        # episodes_progress_bar.close()
        #####################################

        # MAX_STEPS = 100 + 80
        if plot:
            s, a, r, ss, absorb, last = parse_dataset(dataset)
            # axs[0].plot(r[:MAX_STEPS])
            # axs[1].plot(a[:MAX_STEPS])
            axs[0].plot(r)
            axs[1].plot(a)
    if plot:
        if alg == "gp":
            axs[0].set_title(
                f"run policy: {alg} (fastgp={fast_gp}, stoch={stoch_preds})"
            )
        else:
            axs[0].set_title(f"run policy: {alg} (epoch {agent_epoch})")
        axs[0].grid(True)
        axs[1].grid(True)
        axs[1].set_xlabel("steps")
        axs[0].set_ylabel("reward")
        axs[1].set_ylabel("action")
        plt.savefig(
            os.path.join(
                repo_dir, results_dir, f"run_policy_{alg}_ep{agent_epoch}.png"
            ),
            dpi=150,
        )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(render_policy)
