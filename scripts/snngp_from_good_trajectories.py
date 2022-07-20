import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from experiment_launcher import run_experiment
from torch.utils.data import DataLoader

from src.datasets.dataframe_datasets import DataFrameDataset
from src.models.linear_bayesian_models import SpectralNormalizedNeuralGaussianProcess
from src.utils.seeds import fix_random_seed


def experiment(
    alg: str = "snngp",
    dataset_file: str = "../models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_25000.pkl.gz",
    n_datapoints: int = 100,
    n_epochs: int = 5,
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
    df = df.iloc[:n_datapoints, :]
    df = df.astype("float32")
    # x_cols = ["s0", "s1", "s2", "s3", "s4", "s5"]
    x_df = df[["s0"]]
    y_df = df[["a"]]
    dim_in = x_df.shape[1]
    dim_out = y_df.shape[1]

    train_dataset = DataFrameDataset(x_df, y_df, train=True, seed=seed)
    test_dataset = DataFrameDataset(x_df, y_df, train=False, seed=seed)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # # whiten data here for simplicity
    # x_mean, x_std = x_train.mean(), x_train.std()
    # y_mean, y_std = y_train.mean(), y_train.std()
    # x_train = (x_train - x_mean) / x_std
    # x_test = (x_test - x_mean) / x_std
    # y_train = (y_train) / y_std
    # y_test = (y_test) / y_std

    model = SpectralNormalizedNeuralGaussianProcess(dim_in, dim_out, n_features, lr)
    trace = model.train(train_dataloader, n_epochs)

    ####################################################################################################################
    # EVALUATION

    # data for plotting
    x_train, y_train = train_dataset[:]
    x_test, y_test = test_dataset[:]

    # plot train and test data
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_train, y_train, c="grey", marker="x", s=5, label="train")
    ax.scatter(x_test, y_test, c="k", s=5, label="test")
    # plot prediction
    mu_train, sigma_train, _, _ = model(x_train)
    mu_test, sigma_test, _, _ = model(x_test)
    # plt.errorbar(
    #     x_train.reshape(-1),
    #     mu_train.reshape(-1),
    #     torch.diag(sigma_train),
    #     fmt="none",
    #     color="r",
    #     label="pred",
    # )
    plt.errorbar(
        x_test.reshape(-1),
        mu_test.reshape(-1),
        torch.diag(sigma_test),
        fmt="none",
        color="r",
        label="pred",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.legend()
    ax.set_title(f"snngp (N={n_datapoints}, {n_epochs} epochs)")

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
