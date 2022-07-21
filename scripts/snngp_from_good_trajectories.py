import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from experiment_launcher import run_experiment
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.datasets.dataframe_datasets import DataFrameDataset
from src.models.linear_bayesian_models import SpectralNormalizedNeuralGaussianProcess
from src.utils.conversion_utils import df2torch
from src.utils.plotting_utils import plot_gp
from src.utils.seeds import fix_random_seed


def experiment(
    alg: str = "snngp",
    dataset_file: str = "../models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_100trajs.pkl.gz",
    n_trajectories: int = 20,  # 80% train, 20% test
    n_epochs: int = 20,
    batch_size: int = 64,
    n_features: int = 512,
    lr: float = 4e-3,
    # use_cuda: bool = False,
    # verbose: bool = False,
    # model_save_frequency: bool = 5,  # every x epochs
    # log_wandb: bool = True,
    # wandb_project: str = "smbrl",
    # wandb_entity: str = "showmezeplozz",
    # wandb_group: str = "nlm_clone_SAC",
    # wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "../logs/tmp/",
    # debug: bool = True,
):
    ####################################################################################################################
    # SETUP
    time_begin = time.time()

    # Fix seeds
    fix_random_seed(seed)

    # fix path
    this_dir = os.path.dirname(os.path.abspath(__file__))

    ####################################################################################################################
    # EXPERIMENT

    print(f"Alg: {alg}, Seed: {seed}, Dataset: {dataset_file}")

    # df: [s0-s5, a, r, ss0-ss5, absorb, last]
    df = pd.read_pickle(os.path.join(this_dir, dataset_file))
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

    # # whiten data here for simplicity
    # x_mean, x_std = x_train.mean(), x_train.std()
    # y_mean, y_std = y_train.mean(), y_train.std()
    # x_train = (x_train - x_mean) / x_std
    # x_test = (x_test - x_mean) / x_std
    # y_train = (y_train) / y_std
    # y_test = (y_test) / y_std

    train_dataset = DataFrameDataset(train_x_df, train_y_df)
    test_dataset = DataFrameDataset(test_x_df, test_y_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    dim_in = len(x_cols)
    dim_out = len(y_cols)
    model = SpectralNormalizedNeuralGaussianProcess(dim_in, dim_out, n_features, lr)
    trace = model.train(train_dataloader, n_epochs)

    ####################################################################################################################
    # EVALUATION

    # data for plotting
    x_train, y_train = train_dataset[:]
    x_test, y_test = test_dataset[:]

    ## plot statistics over trajectories
    mu_pred, sigma_pred, _, _ = model(x_test)
    test_df["pred"] = mu_pred.reshape(-1)
    # test_df['pred_var'] = torch.diag(sigma_pred).reshape(-1)
    mean_traj = test_df.groupby(level=0).mean()
    std_traj = test_df.groupby(level=0).std()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    MAX_TIME = 100
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

    ## print train and test MAE, MSE, RMSE
    def log_MAE_MSE_RMSE(pred, test, prefix=""):
        MAE = torch.abs(pred - test).mean()
        MSE = torch.pow(pred - test, 2).mean()
        RMSE = torch.sqrt(MSE)
        print(
            f"{prefix}: MAE={MAE.item():.2f}, MSE={MSE.item():.2f}, RMSE={RMSE.item():.2f}"
        )
        return (
            MAE,
            MSE,
        )

    mu_pred_train, sigma_pred_train, _, _ = model(x_train)
    log_MAE_MSE_RMSE(mu_pred_train, y_train, prefix="TRAIN")
    mu_pred, sigma_pred, _, _ = model(x_test)
    log_MAE_MSE_RMSE(mu_pred, y_test, prefix="TEST")

    # # plot labels and predictions over time
    # MAX_TIME = 100  # tweak traj length for plotting
    # # TODO: only 1 traj yet
    # x_df = test_traj_dfs[0][x_cols]
    # y_df = test_traj_dfs[0][y_cols]
    # x_time = torch.tensor(range(0, len(x_df)))[:MAX_TIME]
    # s_data = df2torch(x_df)[:MAX_TIME]
    # a_data = df2torch(y_df)[:MAX_TIME]
    # mu_pred, sigma_pred, _, _ = model(s_data)
    # fig, ax = plt.subplots(1, 1)
    # plt.plot(x_time, a_data.reshape(-1), c="k", label="data")
    # plot_gp(ax, x_time, mu_pred, torch.diag(sigma_pred), label="pred")
    # ax.set_xlabel("time")
    # ax.set_ylabel("a")
    # # ax.set_title("snngp action prediction")
    # ax.set_title(f"snngp ({len(train_traj_dfs)}/{len(test_traj_dfs)} episodes, {n_epochs} epochs)")
    # ax.legend()

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

    # # plot training loss
    # fig_trace, ax_trace = plt.subplots()
    # ax_trace.plot(trace, c="k")
    # # ax_trace.plot(trace, '.', c='k') # dots for every data point
    # ax_trace.set_title("loss")
    # mu_test, sigma_test, _, _ = model(x_test)

    # # plot features on test dataset (sorted for plotting)
    # x_test_sorted, _ = x_test.sort(dim=0)
    # f = model.features(x_test_sorted).detach()
    # fig_features, ax_features = plt.subplots(figsize=(12, 9))
    # ax_features.plot(x_test_sorted, f[:, ::10], alpha=0.25)
    # ax_features.set_title("features on test dataset")

    plt.show()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
