import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.linear_bayesian_models import SpectralNormalizedNeuralGaussianProcess
from src.utils.conversion_utils import df2torch, map_cpu
from src.utils.plotting_utils import plot_gp
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


def experiment(
    alg: str = "snngp",
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_1000trajs_det.pkl.gz",
    n_trajectories: int = 1,  # 80% train, 20% test
    n_epochs: int = 1000,
    batch_size: int = 200,  # max 200 (steps per episode)
    n_features: int = 64,
    lr: float = 4e-3,
    use_cuda: bool = True,
    # verbose: bool = False,
    plotting: bool = False,
    model_save_frequency: bool = -1,  # every x epochs
    # log_wandb: bool = True,
    # wandb_project: str = "smbrl",
    # wandb_entity: str = "showmezeplozz",
    wandb_group: str = "snngp_clone_SAC",
    # wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,
):
    ####################################################################################################################
    # SETUP
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

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    ####################################################################################################################
    # EXPERIMENT

    print(f"Alg: {alg}, Seed: {seed}, Dataset: {dataset_file}")
    print(f"Logs in {results_dir}")

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

    train_x = df2torch(train_x_df).to(device)
    train_y = df2torch(train_y_df).to(device)
    test_x = df2torch(test_x_df).to(device)
    test_y = df2torch(test_y_df).to(device)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    dim_in = len(x_cols)
    dim_out = len(y_cols)
    model = SpectralNormalizedNeuralGaussianProcess(
        dim_in, dim_out, n_features, lr, device=device
    )
    model.with_whitening(train_x, train_y, method="PCA")

    def log_metrics(epoch=-1):
        ## print train and test MAE, MSE, RMSE
        def compute_MAE_MSE_RMSE(pred, test, prefix=""):
            MAE = torch.abs(pred - test).mean()
            MSE = torch.pow(pred - test, 2).mean()
            RMSE = torch.sqrt(MSE)
            return (
                MAE,
                MSE,
                RMSE,
            )

        with open(os.path.join(results_dir, "metrics.txt"), "a") as f:
            x_train, y_train = train_dataset[:]
            x_test, y_test = test_dataset[:]
            mu_pred_train, sigma_pred_train, _, _ = model(x_train)
            MAE, MSE, RMSE = compute_MAE_MSE_RMSE(mu_pred_train, y_train)
            logstring = f"Epoch {epoch} Train: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
            print("\r" + logstring, end=" ")
            f.write(logstring + "\t")
            mu_pred, sigma_pred, _, _ = model(x_test)
            MAE, MSE, RMSE = compute_MAE_MSE_RMSE(mu_pred, y_test)
            logstring = f"Epoch {epoch} Test: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
            print("\r" + logstring, end=" ")
            f.write(logstring + "\n")

    # train
    trace = []
    for n in tqdm(range(n_epochs + 1), position=0):
        for i_minibatch, minibatch in enumerate(
            tqdm(train_dataloader, leave=False, position=1)
        ):
            # copied from linearbayesianmodels.py for logging and model saving convenience
            x, y = minibatch
            model.opt.zero_grad()
            ellh = model.ellh(x, y)
            kl = model.kl()
            loss = -ellh + kl
            loss.backward()
            model.opt.step()
            trace.append(loss.detach().item())

        # log metrics every epoch
        with torch.no_grad():
            # use latest minibatch
            loss_ = loss.detach().item()
            y_pred, _, _, _ = model(x)
            rmse = torch.sqrt(torch.pow(y_pred - y, 2).mean()).item()
            logstring = f"Epoch {n} Train: Loss={loss_:.2}, RMSE={rmse:.2f}"
            print("\r" + logstring + "\033[K")  # \033[K = erase to end of line
            with open(os.path.join(results_dir, "metrics.txt"), "a") as f:
                f.write(logstring)

        if n % model_save_frequency == 0:
            # Save the agent
            torch.save(model.state_dict(), os.path.join(results_dir, f"agent_{n}.pth"))

    # Save the agent after training
    torch.save(model.state_dict(), os.path.join(results_dir, f"agent_end.pth"))

    ####################################################################################################################
    # EVALUATION

    ## plot statistics over trajectories
    mus = []
    for i_minibatch, minibatch in enumerate(test_dataloader):
        x, y = minibatch
        mu_pred, sigma_pred, _, _ = map_cpu(model(x.to(model.device)))
        mus.append(mu_pred)

    mus_pred = torch.cat(mus, dim=0)  # dim=0
    # mu_pred, sigma_pred, _, _ = map_cpu(model(test_x.to(model.device)))
    test_df["pred"] = mu_pred.reshape(-1)
    # test_df['pred_var'] = torch.diag(sigma_pred).reshape(-1)
    mean_traj = test_df.groupby(level=0).mean()
    std_traj = test_df.groupby(level=0).std()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    MAX_TIME = 200
    x_time = torch.tensor(range(0, len(mean_traj)))[:MAX_TIME]
    y_data_mean = df2torch(mean_traj["a"])[:MAX_TIME]
    y_data_std = df2torch(std_traj["a"])[:MAX_TIME]
    y_pred_mu_mean = df2torch(mean_traj["pred"])[:MAX_TIME]
    y_pred_mu_std = df2torch(std_traj["pred"])[:MAX_TIME]
    # y_pred_var_mean = df2torch(mean_traj['pred_var'])[:MAX_TIME]
    # y_pred_var_sqrtmean = torch.sqrt(y_pred_var_mean)
    plot_gp(
        ax,
        x_time,
        y_data_mean,
        y_data_std,
        color="b",
        label="test data mean & std trajs",
    )
    # plot_gp(ax, x_time, y_pred_mu_mean, y_pred_var_sqrtmean, color='c', label="pred mean(mu) & sqrt(mean(sigma))")
    plot_gp(
        ax,
        x_time,
        y_pred_mu_mean,
        y_pred_mu_std,
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
    ax_trace.semilogy(trace, c="k")
    ax_trace.set_xlabel("minibatches")
    twiny = ax_trace.twiny()
    twiny.set_xlabel("epochs")
    twiny.xaxis.set_ticks_position("bottom")
    twiny.xaxis.set_label_position("bottom")
    twiny.spines.bottom.set_position(("axes", -0.2))
    twiny.set_xlim(
        ax_trace.get_xlim()[0] * batch_size / n_trajectories / 200,
        ax_trace.get_xlim()[1] * batch_size / n_trajectories / 200,
    )
    twiny.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_trace.set_title(f"snngp loss (n_trajs={n_trajectories}, lr={lr:.0e})")
    plt.savefig(os.path.join(results_dir, "loss.png"), dpi=150)

    # # plot features on test dataset (sorted for plotting)
    # x_test_sorted, _ = x_test.sort(dim=0)
    # f = model.features(x_test_sorted).detach()
    # fig_features, ax_features = plt.subplots(figsize=(12, 9))
    # ax_features.plot(x_test_sorted, f[:, ::10], alpha=0.25)
    # ax_features.set_title("features on test dataset")

    if plotting:
        plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    print(f"Logs in {results_dir}")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
