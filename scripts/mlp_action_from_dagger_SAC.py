import os
import time

import matplotlib.pyplot as plt
import numpy as np
import quanser_robots
import seaborn as sns
import torch
import yaml
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from mushroom_rl.core import Agent, Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import arrays_as_dataset, compute_J, parse_dataset
from tqdm import tqdm

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.models.dnns import DNN3
from src.utils.conversion_utils import dataset2df_4, np2torch
from src.utils.environment_tools import rollout, state4to6
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "mlp",
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs_det.pkl.gz",
    n_initial_replay_episodes: int = 10,
    max_replay_buffer_size: int = int(1e4),
    n_test_episodes: int = 10,
    n_epochs_between_rollouts: int = 10,  # epochs before dagger rollout & aggregation
    n_epochs: int = 1500,
    batch_size: int = 200 * 10,  # lower if gpu out of memory
    n_features: int = 128,
    lr: float = 1e-3,
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
    wandb_group: str = "mlp_clone_SAC",
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

    x_cols = ["s0", "s1", "s2", "s3", "s4", "s5"]
    y_cols = ["a"]
    dim_in = len(x_cols)
    dim_out = len(y_cols)

    train_buffer = ReplayBuffer(
        dim_in,
        dim_out,
        batchsize=batch_size,
        device=device,
        max_size=max_replay_buffer_size,
    )

    test_buffer = ReplayBuffer(dim_in, dim_out, batchsize=batch_size, device=device)

    ### mdp ###
    with open(os.path.join(sac_results_dir, "args.yaml")) as f:
        sac_args = yaml.load(f, Loader=yaml.Loader)
    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])
    mdp.seed(seed)

    ### sac agent ###
    sac_agent = Agent.load(sac_agent_path)
    core = Core(sac_agent, mdp)

    def sac_policy(state4):
        state6 = state4to6(state4)  # because sac trained on 6dim state
        action_torch = core.agent.policy.compute_action_and_log_prob_t(
            state6, compute_log_prob=False, deterministic=True
        )
        return action_torch.reshape(-1).numpy()

    ### collect train & test data (prefill train rollout buffer) ###
    with torch.no_grad():
        print("Collecting rollouts ...")
        # train buffer
        train_dataset = rollout(
            mdp, sac_policy, n_episodes=n_initial_replay_episodes, show_progress=True
        )
        s, a, r, ss, absorb, last = parse_dataset(train_dataset)  # everything 4dim
        new_xs = np2torch(np.hstack([state4to6(s)]))
        new_ys = np2torch(a)
        train_buffer.add(new_xs, new_ys)
        # test buffer
        test_dataset = rollout(
            mdp, sac_policy, n_episodes=n_test_episodes, show_progress=True
        )
        s, a, r, ss, absorb, last = parse_dataset(test_dataset)  # everything 4dim
        new_xs = np2torch(np.hstack([state4to6(s)]))
        new_ys = np2torch(a)
        test_buffer.add(new_xs, new_ys)
        print("Done.")

    ### agent ###
    model = DNN3(dim_in, dim_out, n_features)
    model.init_whitening(train_buffer.xs, train_buffer.ys, disable_y=True)
    # model.init_whitening(train_buffer.xs, train_buffer.ys)
    model.to(device)

    def model_policy(state4):
        state6_torch = np2torch(state4to6(state4)).to(device)
        action_torch = model(state6_torch)
        return action_torch.cpu().reshape(-1).numpy()

    ### optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ####################################################################################################################
    #### TRAINING

    try:
        losses, test_losses = [], []
        Rs, rmses, bufsizes = [], [], []
        minib_progress = False
        for n in tqdm(range(n_epochs + 1), position=0):
            for i_minibatch, minibatch in enumerate(
                tqdm(train_buffer, leave=False, position=1, disable=not minib_progress)
            ):
                x, y = minibatch
                opt.zero_grad()
                y_pred = model(x)
                loss_f = torch.nn.MSELoss()
                loss = loss_f(y, y_pred)
                loss.backward()
                opt.step()
                losses.append(loss.detach().item())

            if n % n_epochs_between_rollouts == 0:
                with torch.no_grad():
                    # run model against gym
                    dataset = rollout(
                        mdp, model_policy, n_episodes=1, show_progress=False
                    )
                    s, a, r, ss, absorb, last = parse_dataset(
                        dataset
                    )  # everything 4dim
                    # reward of model
                    J = compute_J(dataset, mdp.info.gamma)[0]
                    R = compute_J(dataset, 1.0)[0]
                    new_xs = np2torch(
                        np.hstack([state4to6(s)])
                    )  # take explored states ...
                    a_sac = sac_policy(s).reshape(-1, 1)
                    new_ys = np2torch(a_sac)  # ... and (true) sac actions
                    train_buffer.add(new_xs, new_ys)
                    # store dataset with sac actions
                    modified_dataset = arrays_as_dataset(s, a_sac, r, ss, absorb, last)
                    # logging
                    train_dataset.append(modified_dataset)
                    Rs.append(R)
                    bufsizes.append(train_buffer.size)

            # log metrics
            if n % (n_epochs * log_frequency) == 0:
                # log
                with torch.no_grad():
                    # train rmse
                    with torch.no_grad():
                        _sqr_residuals_sum = []
                        _n_sqr_residuals_ = 0
                        for minibatch in train_buffer:
                            _x, _y = minibatch
                            _y_pred = model(_x)
                            _sqr_residual = torch.pow(_y_pred - _y, 2)
                            _sqr_residuals_sum.append(_sqr_residual.sum().item())
                            _n_sqr_residuals_ += len(_sqr_residual)
                    rmse = np.sqrt(sum(_sqr_residuals_sum) / _n_sqr_residuals_)
                    rmses.append(rmse)

                    # test loss
                    test_buffer.shuffling = False
                    with torch.no_grad():
                        _test_losses = []
                        for minibatch in test_buffer:
                            _x_test, _y_test = minibatch
                            _y_test_pred = model(_x_test.to(device))
                            _test_loss = torch.nn.MSELoss()(_y_test, _y_test_pred)
                            _test_losses.append(_test_loss.item())
                    test_loss = np.mean(_test_losses)
                    test_losses.append(test_loss)

                    logstring = f"Epoch {n} Train: Loss={losses[-1]:.2}"
                    logstring += f", RMSE={rmse:.2f}"
                    logstring += f", test loss={test_losses[-1]:.2}"
                    logstring += f", J={J:.2f}, R={R:.2f}"
                    logstring += f", bufsize={train_buffer.size}"
                    print("\r" + logstring + "\033[K")  # \033[K = erase to end of line
                    with open(os.path.join(results_dir, "metrics.txt"), "a") as f:
                        f.write(logstring + "\n")

            if n % (n_epochs * model_save_frequency) == 0:
                # Save the agent
                torch.save(
                    model.state_dict(), os.path.join(results_dir, f"agent_{n}.pth")
                )
    except KeyboardInterrupt:
        pass  # save policy until now and show results

    # Save the agent after training
    torch.save(model.state_dict(), os.path.join(results_dir, "agent_end.pth"))

    ####################################################################################################################
    #### EVALUATION

    ### plot pointwise action prediction ###
    """pointwise means the model predicts an action for every every state 
    in the dataset. this is explicitly not a rollout, and prediction errors 
    do not add up"""
    s, a, r, ss, absorb, last = parse_dataset(test_dataset)
    _x = np2torch(np.hstack([state4to6(s)]))
    with torch.no_grad():
        a_pred = model(_x.to(device))
        a_pred = a_pred.cpu()
    a = a.reshape(-1, mdp.info.horizon)
    a_pred = a_pred.reshape(-1, mdp.info.horizon)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    x_time = torch.tensor(range(0, 200))
    for ei in range(a.shape[0]):
        ax.plot(x_time, a[ei, :], color="b")
        ax.plot(x_time, a_pred[ei, :], color="r")
    # replot last with label
    ax.plot(x_time, a[ei, :], color="b", label="data")
    ax.plot(x_time, a_pred[ei, :], color="r", label=alg)
    ax.legend()
    ax.set_xlabel("steps")
    ax.set_ylabel("action")
    ax.set_title(
        f"{alg} pointwise action on {n_test_episodes} episodes ({n_epochs} epochs, lr={lr})"
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
        g.fig.suptitle("train data ({len(train_dataset)} episodes)", y=1.01)
        g.savefig(os.path.join(results_dir, "train_data.png"), dpi=150)
        # test data
        df = dataset2df_4(test_dataset)
        # df["traj_id"] = torch.floor(df2torch(df.index) / mdp.info.horizon)
        # g = sns.PairGrid(df[cols], hue="traj_id")
        g = sns.PairGrid(df[cols])
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.figure.suptitle(f"test data ({n_test_episodes} episodes)", y=1.01)
        g.savefig(os.path.join(results_dir, "test_data.png"), dpi=150)

    # plot training stats
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), dpi=150)

    def scaled_xaxis(y_points, n_on_axis):
        return np.arange(len(y_points)) / len(y_points) * n_on_axis

    # plot train & test loss
    axs[0].set_ylabel("loss")
    axs[0].plot(scaled_xaxis(losses, n), losses, c="k", label="train loss")
    axs[0].plot(scaled_xaxis(test_losses, n), test_losses, c="g", label="test loss")
    axs[0].legend()
    axs[0].set_yscale("symlog")
    # plot rmse
    axs[1].plot(scaled_xaxis(rmses, n), rmses, label="rmse")
    axs[1].set_ylim([0, 1.05 * max(rmses)])
    axs[1].set_ylabel("RMSE")
    # plot rewards
    axs[2].plot(scaled_xaxis(Rs, n), Rs, "r", label="cum. reward")
    axs[2].set_ylim([0, 1.05 * max(Rs) + 5])
    axs[2].set_ylabel("cum. reward")
    # plot bufsize
    axs[3].plot(scaled_xaxis(bufsizes, n), bufsizes, "gray", label="buf size")
    axs[3].set_ylabel("replay buffer size")

    axs[3].set_xlabel("epochs")
    axs[0].set_title(f"{alg} training stats (lr={lr:.0e})")
    plt.savefig(os.path.join(results_dir, "training_stats.png"), dpi=150)

    if plotting:
        plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    print(f"Logs in {results_dir}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
