import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quanser_robots
import seaborn as sns
import torch
import yaml
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.models.linear_bayesian_models import NeuralLinearModel
from src.utils.conversion_utils import dataset2df_4, df2torch, np2torch
from src.utils.environment_tools import rollout, state4to6
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "nlm",
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs_det.pkl.gz",
    n_trajectories: int = 5,  # 80% train, 20% test
    n_epochs: int = 500,
    batch_size: int = 200 * 100,  # minibatching iff <= 200*n_traj
    n_features: int = 128,
    lr: float = 5e-4,
    epochs_between_rollouts: int = 10,  # epochs before dagger rollout & aggregation
    det_sac: bool = True,  # if sac used to collect data with dagger is determ.
    use_cuda: bool = True,
    # verbose: bool = False,
    plot_data: bool = False,
    # plot_data: bool = True,
    plotting: bool = False,
    log_frequency: float = 0.01,  # every p% epochs of n_epochs
    model_save_frequency: float = 0.1,  # every n-th of n_epochs
    # log_wandb: bool = True,
    # wandb_project: str = "smbrl",
    # wandb_entity: str = "showmezeplozz",
    wandb_group: str = "nlm_clone_SAC",
    # wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,
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
        repo_dir, results_dir, wandb_group, str(seed), timestamp()
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
    traj_dfs = [
        traj.reset_index(drop=True) for (traj_id, traj) in df.groupby("traj_id")
    ]
    train_traj_dfs, test_traj_dfs = train_test_split(traj_dfs, test_size=0.2)
    train_df = pd.concat(train_traj_dfs)
    test_df = pd.concat(test_traj_dfs)

    x_cols = ["s0", "s1", "s2", "s3", "s4", "s5"]
    y_cols = ["a"]
    dim_in = len(x_cols)
    dim_out = len(y_cols)
    train_x_df, train_y_df = train_df[x_cols], train_df[y_cols]
    test_x_df, test_y_df = test_df[x_cols], test_df[y_cols]

    train_buffer = ReplayBuffer(
        dim_in, dim_out, batchsize=batch_size, device=device, max_size=1e5
    )
    train_buffer.add(df2torch(train_x_df), df2torch(train_y_df))

    test_buffer = ReplayBuffer(
        dim_in, dim_out, batchsize=batch_size, device=device, max_size=1e5
    )
    test_buffer.add(df2torch(test_x_df), df2torch(test_y_df))

    ### mdp ###
    with open(os.path.join(sac_results_dir, "args.yaml")) as f:
        sac_args = yaml.load(f, Loader=yaml.Loader)
    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])
    mdp.seed(seed)

    ### sac agent ###
    sac_agent = Agent.load(sac_agent_path)
    core = Core(sac_agent, mdp)

    def policy(state4):
        state6 = state4to6(state4)  # because sac trained on 6dim state
        action_torch = core.agent.policy.compute_action_and_log_prob_t(
            state6, compute_log_prob=False, deterministic=True
        )
        return action_torch.reshape(-1).numpy()

    ### agent ###
    model = NeuralLinearModel(dim_in, dim_out, n_features)
    # model.init_whitening(train_buffer.xs, train_buffer.ys, disable_y=True)
    model.init_whitening(train_buffer.xs, train_buffer.ys)
    model.to(device)

    ### optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ####################################################################################################################
    #### TRAINING

    train_dataset = []
    loss_trace = []
    test_loss_trace = []
    minib_progress = False
    for n in tqdm(range(n_epochs + 1), position=0):
        for i_minibatch, minibatch in enumerate(
            tqdm(train_buffer, leave=False, position=1, disable=not minib_progress)
        ):
            x, y = minibatch
            opt.zero_grad()
            # ellh = model.ellh(x, y)
            # kl = model.kl()
            # loss = (-ellh + kl) / x.shape[0]
            loss = -model.elbo(x, y)
            loss.backward()
            opt.step()
            loss_trace.append(loss.detach().item())

        if n % epochs_between_rollouts == 0:
            ### collect rollout data ###
            with torch.no_grad():
                dataset = rollout(mdp, policy, n_episodes=1, show_progress=False)
                train_dataset.extend(dataset)
                J = compute_J(dataset, mdp.info.gamma)[0]
                R = compute_J(dataset, 1.0)[0]
                # aggregate
                s, a, r, ss, absorb, last = parse_dataset(dataset)  # everything 4dim
                new_xs = np2torch(np.hstack([state4to6(s)]))
                new_ys = np2torch(a)
                train_buffer.add(new_xs, new_ys)

        # log metrics
        if n % (n_epochs * log_frequency) == 0:
            # log
            with torch.no_grad():
                # use latest minibatch
                y_pred = model(x, covs=False)
                rmse = torch.sqrt(torch.pow(y_pred - y, 2).mean()).item()

                # test loss
                train_buffer.shuffling = False
                test_buffer.shuffling = False
                with torch.no_grad():
                    test_losses = []
                    for minibatch in test_buffer:
                        _x_test, _y_test = minibatch
                        _test_loss = -model.elbo(_x_test, _y_test)
                        test_losses.append(_test_loss.item())
                test_loss = np.mean(test_losses)
                test_loss_trace.append(test_loss)

                logstring = (
                    f"Epoch {n} Train: Loss={loss_trace[-1]:.2}, RMSE={rmse:.2f}"
                )
                logstring += f", test loss={test_loss_trace[-1]:.2}"
                logstring += f", bufsize={train_buffer.size}"
                logstring += f", J={J:.2f}, R={R:.2f}"  # off-sync with computation
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

    ### collect test rollouts ###
    with torch.no_grad():
        print("Collecting test rollouts ...")
        # test buffer
        test_episodes = 10
        test_dataset = rollout(
            mdp, policy, n_episodes=test_episodes, show_progress=True
        )
        s, a, r, ss, absorb, last = parse_dataset(test_dataset)  # everything 4dim
        new_xs = np2torch(np.hstack([state4to6(s)]))
        new_ys = np2torch(a)
        test_buffer.add(new_xs, new_ys)
        print("Done.")

    ### plot pointwise action prediction ###
    """pointwise means the model predicts an action for every every state 
    in the dataset. this is explicitly not a rollout, and prediction errors 
    do not add up"""
    s, a, r, ss, absorb, last = parse_dataset(test_dataset)
    _x = np2torch(np.hstack([state4to6(s)]))
    with torch.no_grad():
        a_pred = model(_x.to(device), covs=False).cpu()
    a = a.reshape(-1, mdp.info.horizon)
    a_pred = a_pred.reshape(-1, mdp.info.horizon)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    x_time = torch.tensor(range(0, 200))
    for ei in range(test_episodes):
        ax.plot(x_time, a[ei, :], color="b")
        ax.plot(x_time, a_pred[ei, :], color="r")
    # replot last with label
    ax.plot(x_time, a[ei, :], color="b", label="data")
    ax.plot(x_time, a_pred[ei, :], color="r", label=alg)
    ax.legend()
    ax.set_xlabel("steps")
    ax.set_ylabel("action")
    ax.set_title(
        f"{alg} pointwise action on {test_episodes} episode ({n_epochs} epochs, lr={lr})"
    )
    plt.savefig(os.path.join(results_dir, "pointwise_action_pred.png"), dpi=150)

    ### plot data space coverage ###
    if plot_data:
        # cols = ["theta", "alpha", "theta_dot", "alpha_dot", "traj_id"]
        cols = ["theta", "alpha", "theta_dot", "alpha_dot"]
        # train data
        df = dataset2df_4(train_dataset)
        # df["traj_id"] = torch.floor(df2torch(df.index) / mdp.info.horizon)
        # g = sns.PairGrid(df[cols], hue="traj_id")
        g = sns.PairGrid(df[cols])
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.fig.suptitle("train data", y=1.01)
        g.savefig(os.path.join(results_dir, "train_data.png"), dpi=150)
        # test data
        df = dataset2df_4(test_dataset)
        # df["traj_id"] = torch.floor(df2torch(df.index) / mdp.info.horizon)
        # g = sns.PairGrid(df[cols], hue="traj_id")
        g = sns.PairGrid(df[cols])
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.figure.suptitle(f"test data ({test_episodes} episodes)", y=1.01)
        g.savefig(os.path.join(results_dir, "test_data.png"), dpi=150)

    # plot training loss
    fig_trace, ax_trace = plt.subplots()

    def scaled_xaxis(y_points, n_on_axis):
        return np.arange(len(y_points)) / len(y_points) * n_on_axis

    x_train_loss = scaled_xaxis(loss_trace, n_epochs)
    ax_trace.plot(x_train_loss, loss_trace, c="k", label="train loss")
    x_test_loss = scaled_xaxis(test_loss_trace, n_epochs)
    ax_trace.plot(x_test_loss, test_loss_trace, c="g", label="test loss")
    ax_trace.set_yscale("symlog")
    ax_trace.set_xlabel("epochs")
    ax_trace.set_ylabel("loss")
    ax_trace.set_title(f"{alg} loss (lr={lr:.0e})")
    fig_trace.legend()
    plt.savefig(os.path.join(results_dir, "loss.png"), dpi=150)

    if plotting:
        plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    print(f"Logs in {results_dir}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
