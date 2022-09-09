import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
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
    dataset_file: str = "models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs.pkl.gz",
    n_trajectories: int = 100,  # 80% train, 20% test
    n_epochs: int = -1,
    batch_size: int = 64,
    n_features: int = 512,
    lr: float = 4e-3,
    use_cuda: bool = True,
    # verbose: bool = False,
    plotting: bool = True,
    model_save_frequency: bool = 5,  # every x epochs
    # log_wandb: bool = True,
    # wandb_project: str = "smbrl",
    # wandb_entity: str = "showmezeplozz",
    wandb_group: str = "snngp_learn_dynamics",
    # wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,
    yid: str = "5",  # WIP: train for this next-state id, later train on all
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

    # df: [s0-s5, a, r, ss0-ss5, absorb, last]
    df = pd.read_pickle(os.path.join(repo_dir, dataset_file))
    df = df[df["traj_id"] < n_trajectories]  # take only n_trajectories
    df = df.astype("float32")  # otherwise spectral_norm complains
    # add state deltas
    for i in range(6):
        df[f"ds{i}"] = df[f"ss{i}"] - df[f"s{i}"]
    traj_dfs = [
        traj.reset_index(drop=True) for (traj_id, traj) in df.groupby("traj_id")
    ]
    train_traj_dfs, test_traj_dfs = train_test_split(traj_dfs, test_size=0.2)
    x_cols = ["a", "s0", "s1", "s2", "s3", "s4", "s5"]
    y_cols = [f"ds{yid}"]  # WIP: later train on all outputs
    # y_cols = ["ds0", "ds1"]
    # y_cols = ["ds0", "ds1", "ds2", "ds3", "ds4", "ds5"]
    train_df = pd.concat(train_traj_dfs)
    test_df = pd.concat(test_traj_dfs)
    train_x_df, train_y_df = train_df[x_cols], train_df[y_cols]
    test_x_df, test_y_df = test_df[x_cols], test_df[y_cols]

    # # whiten data here for simplicity
    # x_mean, x_std = x_train.mean(), x_train.std()
    # y_mean, y_std = y_train.mean(), y_train.std()
    # x_train = (x_train - x_mean) / x_std
    # x_test = (x_test - x_mean) / x_std
    # y_train = (y_train) / y_std
    # y_test = (y_test) / y_std

    train_dataset = TensorDataset(
        df2torch(train_x_df).to(device), df2torch(train_y_df).to(device)
    )
    test_dataset = TensorDataset(
        df2torch(test_x_df).to(device), df2torch(test_y_df).to(device)
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    dim_in = len(x_cols)
    dim_out = len(y_cols)
    model = SpectralNormalizedNeuralGaussianProcess(
        dim_in, dim_out, n_features, lr, device=device
    )

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

        if n % model_save_frequency == 0:
            # Save the agent
            torch.save(model.state_dict(), os.path.join(results_dir, f"agent_{n}.pth"))

    # Save the agent after training
    torch.save(model.state_dict(), os.path.join(results_dir, f"agent_end.pth"))

    ####################################################################################################################
    # EVALUATION

    def compute_MAE_MSE_RMSE(pred, test, average=True, prefix=""):
        if average:
            MAE = torch.abs(pred - test).mean()
            MSE = torch.pow(pred - test, 2).mean()
        else:
            MAE = torch.abs(pred - test)
            MSE = torch.pow(pred - test, 2)
        RMSE = torch.sqrt(MSE)
        return (
            MAE,
            MSE,
            RMSE,
        )

    # data for plotting
    x_train, y_train = train_dataset[:]
    x_test, y_test = test_dataset[:]

    # s-a plot error of next state prediction
    mu_pred, sigma_pred, _, _ = map_cpu(model(x_test.to(model.device)))
    test_df[f"ds{yid}pred"] = mu_pred.reshape(-1)
    test_df[f"ss{yid}pred"] = test_df[f"s{yid}"] + test_df[f"ds{yid}pred"]
    MAE, MSE, RMSE = compute_MAE_MSE_RMSE(
        df2torch(test_df[f"ss{yid}pred"]), df2torch(test_df[f"ss{yid}"]), average=False
    )
    test_df["std"] = sigma_pred.diag()
    test_df["std"].describe()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    MAX_TIME = 15000
    cmap = mpl.cm.cividis_r
    vmin = RMSE.min()
    vmax = RMSE.max()
    ax.scatter(
        test_df[f"s{yid}"].iloc[:MAX_TIME],
        test_df["a"].iloc[:MAX_TIME],
        s=2e2 * sigma_pred.diag()[:MAX_TIME],
        # s=0.8,
        c=RMSE[:MAX_TIME],
        cmap="cividis_r",
        vmin=vmin,
        vmax=vmax,
        # edgecolors='k',
        alpha=0.7,
    )
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="vertical",
        label=f"RMSE of ss{yid} prediction",
    )

    # get legend size points (https://stackoverflow.com/a/43191354)
    # here we step over the sorted data into 4 or 5 strides and select the
    # last 3 steps as a representative sample, this only works if your
    # data is fairly uniformly distributed
    all_sizes = 2e2 * sigma_pred.diag()[:MAX_TIME]
    _, all_indices = all_sizes.sort()
    indices = all_indices[:: len(all_sizes) // 4][-3:]
    # legend_sizes = [::len(all_sizes)//4][-3:]
    # get the indices for each of the legend sizes
    # indices = [torch.where(all_sizes==v)[0][0].item() for v in legend_sizes]
    # plot each point again, and its value as a label
    for i in indices:
        i = i.item()
        ax.scatter(
            test_df[f"s{yid}"].iloc[i],
            test_df["a"].iloc[i],
            s=2e2 * sigma_pred.diag()[i],
            c=RMSE[i],
            cmap="cividis_r",
            # vmin=vmin,
            # vmax=vmax,
            # edgecolors='k',
            alpha=0.7,
            label=f"var = {sigma_pred.diag()[i]:.2f}",
        )
    # add the legend
    ax.legend(scatterpoints=1)

    ax.set_xlabel(f"s{yid}")
    ax.set_ylabel("a")
    ax.set_title(
        f"snngp ss{yid} ({len(train_traj_dfs)}/{len(test_traj_dfs)} episodes, {n_epochs} epochs)"
    )
    plt.show()
    plt.savefig(
        os.path.join(
            results_dir,
            f"ss{yid}__state-action-nextstate_scatter_plot__test_data_pred_error_and_pred_unc.png",
        ),
        dpi=150,
    )

    ## plot statistics over trajectories
    mu_pred, sigma_pred, _, _ = map_cpu(model(x_test.to(model.device)))
    test_df[f"ds{yid}pred"] = mu_pred.reshape(-1)
    # test_df['pred_var'] = torch.diag(sigma_pred).reshape(-1)
    mean_traj = test_df.groupby(level=0).mean()
    std_traj = test_df.groupby(level=0).std()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    MAX_TIME = 100
    x_time = torch.tensor(range(0, len(mean_traj)))[:MAX_TIME]
    y_data_mean = df2torch(mean_traj[f"ds{yid}"])[:MAX_TIME]
    y_data_std = df2torch(std_traj[f"ds{yid}"])[:MAX_TIME]
    y_pred_mu_mean = df2torch(mean_traj[f"ds{yid}pred"])[:MAX_TIME]
    y_pred_mu_std = df2torch(std_traj[f"ds{yid}pred"])[:MAX_TIME]
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
    ax.set_ylabel(f"ds{yid}")
    ax.set_title(
        f"snngp ds{yid} ({len(train_traj_dfs)}/{len(test_traj_dfs)} episodes, {n_epochs} epochs)"
    )
    ax.legend()
    plt.savefig(
        os.path.join(results_dir, "mean-std_traj_plots__test_data_pred.png"), dpi=150
    )

    ## print train and test MAE, MSE, RMSE
    with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
        mu_pred_train, sigma_pred_train, _, _ = model(x_train)
        MAE, MSE, RMSE = compute_MAE_MSE_RMSE(mu_pred_train, y_train)
        logstring = (
            f"Train: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
        )
        print(logstring)
        f.write(logstring + "\n")
        mu_pred, sigma_pred, _, _ = model(x_test)
        MAE, MSE, RMSE = compute_MAE_MSE_RMSE(mu_pred, y_test)
        logstring = (
            f"Test: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
        )
        print(logstring)
        f.write(logstring + "\n")

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
    ax_trace.plot(trace, c="k")
    # ax_trace.plot(trace, '.', c='k') # dots for every data point
    ax_trace.set_xlabel("minibatches")
    ax_trace.set_title("loss")
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
