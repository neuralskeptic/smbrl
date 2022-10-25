import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quanser_robots
import torch
import yaml
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from matplotlib.ticker import MaxNLocator
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.models.linear_bayesian_models import SpectralNormalizedNeuralGaussianProcess
from src.utils.conversion_utils import df2torch, np2torch
from src.utils.plotting_utils import plot_gp
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "snngp",
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs_det.pkl.gz",
    n_trajectories: int = 5,  # 80% train, 20% test
    n_epochs: int = 1000,
    batch_size: int = 200 * 100,  # minibatching iff <= 200*n_traj
    n_features: int = 128,
    lr: float = 1e-4,
    epochs_between_rollouts: int = 10,  # epochs before dagger rollout & aggregation
    use_cuda: bool = True,
    # verbose: bool = False,
    plotting: bool = False,
    log_frequency: float = 0.01,  # every p% epochs of n_epochs
    model_save_frequency: float = 0.1,  # every n-th of n_epochs
    # log_wandb: bool = True,
    # wandb_project: str = "smbrl",
    # wandb_entity: str = "showmezeplozz",
    wandb_group: str = "snngp_learn_dynamics",
    # wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,
    yid: int = 0,  # WIP: train for this next-state id, later train on all
):
    ####################################################################################################################
    #### SETUP (saved to yaml)
    time_begin = time.time()

    if debug:
        # disable wandb logging and redirect normal logging to ./debug directory
        print("@@@@@@@@@@@@@@@@@ DEBUG: LOGGING DISABLED @@@@@@@@@@@@@@@@@")
        os.environ["WANDB_MODE"] = "disabled"
        results_dir = os.path.join("debug", results_dir)

    # Fix seeds
    fix_random_seed(seed)

    # Results directory
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
    results_dir = os.path.join(
        repo_dir, results_dir, wandb_group, str(seed), f"s{yid}", timestamp()
    )
    os.makedirs(results_dir, exist_ok=True)

    # sac agent dir
    sac_results_dir = os.path.dirname(os.path.join(repo_dir, dataset_file))
    sac_agent_path = os.path.join(repo_dir, sac_results_dir, "agent_end.msh")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    ####################################################################################################################
    #### EXPERIMENT SETUP

    print(f"Alg: {alg}, Seed: {seed}, Dataset: {dataset_file}")
    print(f"Logs in {results_dir}")

    ### INITIAL DATASET ###
    # df: [s0-s5, a, r, ss0-ss5, absorb, last]
    df = pd.read_pickle(os.path.join(repo_dir, dataset_file))
    df = df[df["traj_id"] < n_trajectories]  # take only n_trajectories
    df = df.astype("float32")  # otherwise spectral_norm complains
    ##### qube.step(a) ##### (self._state = th, al, thd, ald)
    # rwd, done = self._rwd(self._state, a)
    # self._state, act = self._ctrl_step(a)
    # obs = np.float32([np.cos(self._state[0]), np.sin(self._state[0]),
    #                   np.cos(self._state[1]), np.sin(self._state[1]),
    #                   self._state[2], self._state[3]])
    # return obs, rwd, done, {'s': self._state, 'a': act}
    ########################
    # transform state space: sin/cos to angles (atan2 approx averages errors)
    df["_s0"] = np.arctan2(df["s1"], df["s0"])
    df["_s1"] = np.arctan2(df["s3"], df["s2"])
    df["_s2"] = df["s4"]
    df["_s3"] = df["s5"]
    df["_ss0"] = np.arctan2(df["ss1"], df["ss0"])
    df["_ss1"] = np.arctan2(df["ss3"], df["ss2"])
    df["_ss2"] = df["ss4"]
    df["_ss3"] = df["ss5"]

    def state4to6(state4):
        state6 = np.empty(6)
        state6[0] = np.cos(state4[0])
        state6[1] = np.sin(state4[0])
        state6[2] = np.cos(state4[1])
        state6[3] = np.sin(state4[1])
        state6[4] = state[2]
        state6[5] = state[3]
        return state6

    # add state deltas
    for i in range(4):
        df[f"_ds{i}"] = df[f"_ss{i}"] - df[f"_s{i}"]
    traj_dfs = [
        traj.reset_index(drop=True) for (traj_id, traj) in df.groupby("traj_id")
    ]
    train_traj_dfs, test_traj_dfs = train_test_split(traj_dfs, test_size=0.2)
    train_df = pd.concat(train_traj_dfs)
    test_df = pd.concat(test_traj_dfs)

    x_cols = ["_s0", "_s1", "_s2", "_s3", "a"]
    y_cols = [f"_ds{yid}"]  # WIP: later train on all outputs
    dim_in = len(x_cols)
    dim_out = len(y_cols)
    train_x_df, train_y_df = train_df[x_cols], train_df[y_cols]
    test_x_df, test_y_df = test_df[x_cols], test_df[y_cols]

    test_x = df2torch(test_x_df).to(device)
    test_y = df2torch(test_y_df).to(device)

    test_dataset = TensorDataset(test_x, test_y)

    # TODO replace test dataloader with ReplayBuffer
    test_dataloader_shuf = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_buffer = ReplayBuffer(dim_in, dim_out, batchsize=batch_size, device=device)
    train_buffer.add(df2torch(train_x_df), df2torch(train_y_df))

    ### mdp ###
    try:
        with open(os.path.join(sac_results_dir, "args.json")) as f:
            sac_args = json.load(f)
    except:
        with open(os.path.join(sac_results_dir, "args.yaml")) as f:
            sac_args = yaml.load(f, Loader=yaml.Loader)
    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])

    ### exploration agent ###
    sac_agent = Agent.load(sac_agent_path)
    prepro = None
    core = Core(sac_agent, mdp)

    ### snngp agent ###
    model = SpectralNormalizedNeuralGaussianProcess(
        dim_in, dim_out, n_features, lr, device=device
    )
    model.with_whitening(train_buffer.states, train_buffer.actions)

    ####################################################################################################################
    #### TRAINING

    loss_trace = []
    one_minib = False
    if batch_size == n_trajectories * 200 or n_trajectories < 150:
        one_minib = True
    for n in tqdm(range(n_epochs + 1), position=0):
        for i_minibatch, minibatch in enumerate(
            tqdm(train_buffer, leave=False, position=1, disable=one_minib)
        ):
            x, y = minibatch
            model.opt.zero_grad()
            ellh = model.ellh(x, y)
            kl = model.kl()
            loss = (-ellh + kl) / x.shape[0]
            loss.backward()
            model.opt.step()
            loss_trace.append(loss.detach().item())

        if n % epochs_between_rollouts == 0:
            # run exploration policy against gym (on cpu)
            with torch.no_grad():
                dataset = list()
                episode_steps = 0
                last = False
                mdp.reset(None)  # random initial state
                _, _, _, ssa_dict = mdp.step(np.zeros(1))  # step once to get state4
                state = ssa_dict["s"].copy()

                while not last:
                    # choose exploratory action
                    state6 = state4to6(state)  # because sac trained on 6dim state
                    if torch.randn(1) > 0.5:  # SAC det
                        action_torch = core.agent.policy.compute_action_and_log_prob_t(
                            state6, compute_log_prob=False, deterministic=True
                        )
                    else:  # SAC stoch
                        action_torch = core.agent.policy.compute_action_and_log_prob_t(
                            state6, compute_log_prob=False, deterministic=False
                        )
                    action = action_torch.reshape(-1).numpy()
                    # run action in env
                    next_state_6, reward, absorbing, ssa_dict = mdp.step(action)
                    next_state = ssa_dict["s"]  # angle, not cos/sin
                    episode_steps += 1
                    if episode_steps >= mdp.info.horizon or absorbing:
                        last = True
                    sample = (state, action, reward, next_state, absorbing, last)
                    dataset.append(sample)
                    state = next_state.copy()
                # J = compute_J(dataset, mdp.info.gamma)[0]
                # R = compute_J(dataset, 1.0)[0]

                # aggregate data
                new_xs = torch.empty((len(dataset), dim_in))
                new_ys = torch.empty((len(dataset), dim_out))
                for i in range(len(dataset)):
                    new_state = np2torch(dataset[i][0])
                    new_action = np2torch(dataset[i][1])
                    new_next_state = np2torch(dataset[i][3])
                    new_delta_state = new_next_state - new_state
                    new_xs[i, :] = torch.cat((new_action, new_state)).reshape(-1)
                    new_ys[i, :] = new_delta_state[yid].reshape(-1)
                train_buffer.add(new_xs, new_ys)

        # log metrics every epoch
        if n % (n_epochs * log_frequency) == 0:
            # print(f"GPU MEM LEAK: {torch.cuda.memory_allocated():,} B".replace(",", "_"))
            # log
            with torch.no_grad():
                # use latest minibatch
                loss_ = loss_trace[-1]
                y_pred, _, _, _ = model(x)
                rmse = torch.sqrt(torch.pow(y_pred - y, 2).mean()).item()
                logstring = f"Epoch {n} Train: Loss={loss_:.2}, RMSE={rmse:.2f}"
                # logstring += f", J={J:.2f}, R={R:.2f}"  # off-sync with computation
                print("\r" + logstring + "\033[K")  # \033[K = erase to end of line
                with open(os.path.join(results_dir, "metrics.txt"), "a") as f:
                    f.write(logstring + "\n")

        if n % (n_epochs * model_save_frequency) == 0:
            # Save the agent
            torch.save(model.state_dict(), os.path.join(results_dir, f"agent_{n}.pth"))

    mdp.stop()

    # Save the agent after training
    torch.save(model.state_dict(), os.path.join(results_dir, "agent_end.pth"))

    ####################################################################################################################
    #### EVALUATION

    ## plot statistics over trajectories
    y_pred_list = []
    for i_minibatch, minibatch in enumerate(test_dataloader):
        x, _ = minibatch
        y_pred, _, _, _ = model(x.to(model.device))
        y_pred_list.append(y_pred.cpu())

    y_pred = torch.cat(y_pred_list, dim=0)  # dim=0

    x_time = torch.tensor(range(0, 200))
    y_data_mean = test_y.reshape(-1, 200).mean(dim=0)
    y_data_std = test_y.reshape(-1, 200).std(dim=0)
    y_pred_mu_mean = y_pred.reshape(-1, 200).mean(dim=0)
    y_pred_mu_std = y_pred.reshape(-1, 200).std(dim=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plot_gp(
        ax,
        x_time,
        y_data_mean.cpu(),
        y_data_std.cpu(),
        color="b",
        label="test data mean & std trajs",
    )
    # plot_gp(ax, x_time, y_pred_mu_mean, y_pred_var_sqrtmean, color='c', label="pred mean(mu) & sqrt(mean(sigma))")
    plot_gp(
        ax,
        x_time,
        y_pred_mu_mean.cpu(),
        y_pred_mu_std.cpu(),
        color="r",
        alpha=0.2,
        label="test pred mean(mu) & std(mu)",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("a")
    ax.set_title(
        f"snngp ({len(train_traj_dfs)}/{len(test_traj_dfs)} episodes, {n_epochs} epochs)"
    )
    ax.legend()
    plt.savefig(
        os.path.join(results_dir, "mean-std_traj_plots__test_data_pred.png"), dpi=150
    )

    # # plot train and test data and prediction
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(x_train, y_train, c="grey", marker="x", s=5, label="train")
    # ax.scatter(x_test, y_test, c="k", s=5, label="test")
    # # mu_train, sigma_train, _, _ = model(x_train)
    # # plt.errorbar(
    # #     x_train.reshape(-1),
    # #     mu_train.reshape(-1),
    # #     torch.diag(sigma_train),
    # #     fmt="none",
    # #     color="r",
    # #     label="pred",
    # # )
    # mu_test, sigma_test, _, _ = model(x_test)
    # plt.errorbar(
    #     x_test.reshape(-1),
    #     mu_test.reshape(-1),
    #     torch.diag(sigma_test),
    #     fmt="none",
    #     color="r",
    #     label="pred",
    # )
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.legend()
    # ax.set_title(f"snngp (N={n_datapoints}, {n_epochs} epochs)")

    # plot training loss
    fig_trace, ax_trace = plt.subplots()
    ax_trace.plot(loss_trace, c="k")
    ax_trace.set_yscale("symlog")
    ax_trace.set_xlabel("minibatches")
    twiny = ax_trace.twiny()
    twiny.set_xlabel("epochs")
    twiny.xaxis.set_ticks_position("bottom")
    twiny.xaxis.set_label_position("bottom")
    twiny.spines.bottom.set_position(("axes", -0.2))
    effective_batch_size = min(batch_size, 200 * n_trajectories)
    twiny.set_xlim(
        ax_trace.get_xlim()[0] * effective_batch_size / n_trajectories / 200,
        ax_trace.get_xlim()[1] * effective_batch_size / n_trajectories / 200,
    )
    twiny.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_trace.set_title(f"snngp loss (n_trajs={n_trajectories}, lr={lr:.0e})")
    plt.savefig(os.path.join(results_dir, "loss.png"), dpi=150)

    if plotting:
        plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    print(f"Logs in {results_dir}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
