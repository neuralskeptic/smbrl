import json
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
from mushroom_rl.utils.dataset import parse_dataset, select_first_episodes
from tqdm import tqdm

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.models.linear_bayesian_models import NeuralLinearModel
from src.utils.conversion_utils import dataset2df_4, np2torch
from src.utils.environment_tools import rollout, state4to6, state6to4
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "nlm",
    sac_agent_dir: str = "models/2022_07_15__14_57_42",
    n_train_episodes: int = 100,
    n_epochs: int = 100,
    batch_size: int = 200 * 10,  # lower if gpu out of memory
    n_features: int = 256,
    lr: float = 5e-3,
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
    wandb_group: str = "nlm_learn_dynamics",
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
    sac_results_dir = os.path.join(repo_dir, sac_agent_dir)
    sac_agent_path = os.path.join(repo_dir, sac_results_dir, "agent_end.msh")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    ####################################################################################################################
    #### EXPERIMENT SETUP

    print(f"Alg: {alg}, Seed: {seed}")
    print(f"Logs in {results_dir}")

    x_cols = ["s0", "s1", "s2", "s3", "s4", "s5", "a"]
    y_cols = ["_ds0", "_ds1", "_ds2", "_ds3"]
    dim_in = len(x_cols)
    dim_out = len(y_cols)

    train_buffer = ReplayBuffer(
        dim_in, dim_out, batchsize=batch_size, device=device, max_size=1e5
    )

    test_buffer = ReplayBuffer(
        dim_in, dim_out, batchsize=batch_size, device=device, max_size=1e5
    )

    ### mdp ###
    with open(os.path.join(sac_results_dir, "args.yaml")) as f:
        sac_args = yaml.load(f, Loader=yaml.Loader)
    mdp = Gym(sac_args["env_id"], horizon=sac_args["horizon"], gamma=sac_args["gamma"])
    mdp.seed(seed)

    ### exploration policy (for rollouts) ###
    sac_agent = Agent.load(sac_agent_path)
    core = Core(sac_agent, mdp)

    def policy(state4):
        state6 = state4to6(state4)  # because sac trained on 6dim state
        p = 0.5
        if torch.randn(1) > p:
            # # SAC det
            # action_torch = core.agent.policy.compute_action_and_log_prob_t(
            #     state6, compute_log_prob=False, deterministic=True
            # )
            # gaussian policy (zero mean, 1.5 std)
            action_torch = torch.normal(torch.zeros(1), 1.5 * torch.ones(1))
        else:
            # SAC stoch
            action_torch = core.agent.policy.compute_action_and_log_prob_t(
                state6, compute_log_prob=False, deterministic=False
            )
        return action_torch.reshape(-1).numpy()

    ### collect train data (fill rollout buffer) ###
    with torch.no_grad():
        print("Collecting rollouts ...")
        # train buffer
        train_episodes = n_train_episodes
        train_dataset = rollout(
            mdp, policy, n_episodes=train_episodes, show_progress=True
        )
        s, a, r, ss, absorb, last = parse_dataset(train_dataset)  # everything 4dim
        new_xs = np2torch(np.hstack([state4to6(s), a]))
        new_ys = np2torch(ss - s)  # delta state4 = ss4 - s4
        train_buffer.add(new_xs, new_ys)
        # test buffer
        test_episodes = int(train_episodes / 4)
        test_dataset = rollout(
            mdp, policy, n_episodes=test_episodes, show_progress=True
        )
        s, a, r, ss, absorb, last = parse_dataset(train_dataset)  # everything 4dim
        new_xs = np2torch(np.hstack([state4to6(s), a]))
        new_ys = np2torch(ss - s)  # delta state4 = ss4 - s4
        test_buffer.add(new_xs, new_ys)
        print("Done.")

    ### agent ###
    model = NeuralLinearModel(dim_in, dim_out, n_features)
    model.init_whitening(train_buffer.xs, train_buffer.ys, disable_y=True)
    model.to(device)

    ### optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ####################################################################################################################
    #### TRAINING

    loss_trace = []
    test_loss_trace = []
    minib_progress = False
    for n in tqdm(range(n_epochs + 1), position=0):
        for i_minibatch, minibatch in enumerate(
            tqdm(train_buffer, leave=False, position=1, disable=not minib_progress)
        ):
            x, y = minibatch
            opt.zero_grad()
            loss = -model.elbo(x, y)
            loss.backward()
            opt.step()
            loss_trace.append(loss.detach().item())

        # log metrics
        if n % (n_epochs * log_frequency) == 0:
            # log
            with torch.no_grad():
                # use latest minibatch
                y_pred = model(x, covs=False)
                rmse = torch.sqrt(torch.pow(y_pred - y, 2).mean()).item()

                # test loss
                train_buffer.shuffling = False
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
                print("\r" + logstring + "\033[K")  # \033[K = erase to end of line
                with open(os.path.join(results_dir, "metrics.txt"), "a") as f:
                    f.write(logstring + "\n")

        if n % (n_epochs * model_save_frequency) == 0:
            # Save the agent
            torch.save(model.state_dict(), os.path.join(results_dir, f"agent_{n}.pth"))

    # Save the agent after training
    torch.save(model.state_dict(), os.path.join(results_dir, "agent_end.pth"))

    ####################################################################################################################
    #### EVALUATION

    ### plot pointwise transition prediction (1 episode) ###
    """pointwise means the model predicts a next_state for every every state,action 
    from in the episode from the dataset. this is explicitly not a rollout, 
    and prediction errors do not add up"""
    # s, a, r, ss, absorb, last = select_first_episodes(train_dataset, 1, parse=True)
    s, a, r, ss, absorb, last = select_first_episodes(test_dataset, 1, parse=True)
    _x = np2torch(np.hstack([state4to6(s), a]))
    with torch.no_grad():
        ss_delta_pred = model(_x.to(device), covs=False)
        ss_delta_pred = ss_delta_pred.cpu()
    ss_pred = np2torch(s) + ss_delta_pred

    fig, axs = plt.subplots(4, 2, figsize=(10, 7))
    x_time = torch.tensor(range(0, 200))
    for yi in range(dim_out):
        # delta next state prediction
        axs[yi, 0].plot(x_time, (ss - s)[:, yi], color="b", label="data")
        axs[yi, 0].plot(x_time, ss_delta_pred[:, yi], color="r", label=alg)
        axs[yi, 0].set_ylabel(f"delta ss[{yi}]")
        # next state prediction
        axs[yi, 1].plot(x_time, ss[:, yi], color="b", label="data")
        axs[yi, 1].plot(x_time, ss_pred[:, yi], color="r", label=alg)
        axs[yi, 1].set_ylabel(f"ss[{yi}]")
    axs[0, 1].legend()
    axs[yi, 0].set_xlabel("steps")
    axs[yi, 1].set_xlabel("steps")
    axs[0, 0].set_title("delta state predictions")
    axs[0, 1].set_title("states with predicted delta states")
    fig.suptitle(
        f"{alg} pointwise dynamics on 1 episode ({n_train_episodes} episodes, {n_epochs} epochs, lr={lr})"
    )
    plt.savefig(os.path.join(results_dir, "pointwise_dynamics_pred.png"), dpi=150)

    ### plot rollout of action traj (1 episode) ###
    # s, a, r, ss, absorb, last = select_first_episodes(train_dataset, 1, parse=True)
    s, a, r, ss, absorb, last = select_first_episodes(test_dataset, 1, parse=True)

    open_loop_dataset = list()
    episode_steps = 0
    last = False
    action_iter = iter(a)
    state4 = state6to4(mdp.reset(None).copy())
    while last is False:
        action = action_iter.__next__()
        # dynamics model
        _x = np2torch(np.hstack([state4to6(state4), action]))
        with torch.no_grad():
            ss_delta_pred = model(_x.to(device), covs=False)
        next_state4 = state4 + ss_delta_pred.cpu().numpy()
        episode_steps += 1
        if episode_steps >= mdp.info.horizon:
            last = True
        # get reward from internal gym method
        reward, absorbing = mdp.env.unwrapped._rwd(next_state4, action)
        sample = (state4, action, reward, next_state4, absorbing, last)
        open_loop_dataset.append(sample)
        state4 = next_state4.copy()

    _s, _a, _r, _ss, _absorb, _last = parse_dataset(open_loop_dataset)

    fig, axs = plt.subplots(5, 2, figsize=(10, 7))
    x_time = torch.tensor(range(0, 200))
    # plot rewards
    axs[0, 0].plot(x_time, r, color="b", label="gym")
    axs[0, 0].plot(x_time, _r, color="r", label=alg)
    axs[0, 0].set_ylabel("reward")
    axs[0, 0].legend()
    # plot actions
    axs[0, 1].plot(x_time, a, color="b", label="gym")
    axs[0, 1].set_ylabel("action")
    for yi in range(dim_out):
        # delta next state prediction
        axs[yi + 1, 0].plot(x_time, (ss - s)[:, yi], color="b", label="data")
        axs[yi + 1, 0].plot(x_time, (_ss - _s)[:, yi], color="r", label=alg)
        axs[yi + 1, 0].set_ylabel(f"delta ss[{yi}]")
        # next state prediction
        axs[yi + 1, 1].plot(x_time, ss[:, yi], color="b", label="data")
        axs[yi + 1, 1].plot(x_time, _ss[:, yi], color="r", label=alg)
        axs[yi + 1, 1].set_ylabel(f"ss[{yi}]")
    axs[-1, 0].set_xlabel("steps")
    axs[-1, 1].set_xlabel("steps")
    fig.suptitle(
        f"{alg} action rollout on 1 episode ({n_train_episodes} episodes, {n_epochs} epochs, lr={lr})"
    )
    plt.savefig(os.path.join(results_dir, "action_rollout.png"), dpi=150)

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

    # ## plot buffer
    # buf_pred_list = []
    # train_buffer.shuffling = False
    # for minibatch in train_buffer:
    #     x, _ = minibatch
    #     with torch.no_grad():
    #         y_pred = model(x.to(model.device))
    #     buf_pred_list.append(y_pred.cpu())

    # buf_y_pred = torch.cat(buf_pred_list, dim=0)  # dim=0
    # buf_y_data = train_buffer.ys
    # # buf_pred_trajs = buf_y_pred.reshape(-1, 200)  # unwhitened
    # # buf_data_trajs = buf_y_data.reshape(-1, 200)
    # buf_pred_trajs = whitening.dewhitenY(buf_y_pred.reshape(-1, 200))
    # buf_data_trajs = whitening.dewhitenY(buf_y_data.reshape(-1, 200))

    # fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # for i in range(buf_data_trajs.shape[0]):
    #     i = 0
    #     plt.plot(x_time, buf_data_trajs[i, :].cpu(), color="b")
    #     # plt.plot(x_time, data_trajs[i, :].cpu())
    #     plt.plot(x_time, buf_pred_trajs[i, :].cpu(), color="r")
    #     # plt.plot(x_time, pred_trajs[i, :].cpu())
    # ax.set_xlabel("steps")
    # ax.set_ylabel(y_cols[0])
    # ax.set_title(
    #     f"{alg} on rollout buffer ({buf_data_trajs.shape[0]} episodes, {n_epochs} epochs)"
    # )
    # ax.legend()
    # plt.savefig(os.path.join(results_dir, "traj_plots__test_data_pred.png"), dpi=150)

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
    ax_trace.set_title(f"{alg} loss (n_trajs={n_train_episodes}, lr={lr:.0e})")
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
