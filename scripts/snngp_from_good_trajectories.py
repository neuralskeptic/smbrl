import os
import time

import matplotlib.pyplot as plt
import torch
from experiment_launcher import run_experiment
from torch.utils.data import DataLoader

from src.datasets import toy_datasets
from src.models.linear_bayesian_models import (
    NeuralLinearModel,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.utils.seeds import fix_random_seed


def experiment(
    alg: str = "snngp",
    env_id: str = "Qube-100-v0",
    dataset_file: str = "../models/2022_07_15__14_57_42/SAC_on_Qube-100-v0_25000.pkl.gz",
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
    seed: int = 1234,
    results_dir: str = "../logs/tmp/",
    # debug: bool = True,
):
    ####################################################################################################################
    # SETUP
    time_begin = time.time()

    # Fix seeds
    fix_random_seed(seed)

    def plot_gp(axis, x, mu, var):
        axis.plot(x, mu, "b-")
        for n_std in range(1, 3):
            std = n_std * torch.sqrt(var.squeeze())
            mu = mu.squeeze()
            upper, lower = mu + std, mu - std
            axis.fill_between(
                x.squeeze(),
                upper.squeeze(),
                lower.squeeze(),
                where=upper > lower,
                color="b",
                alpha=0.3,
            )

    ####################################################################################################################
    # EXPERIMENT

    print(f"Alg: {alg}, Seed: {seed}, Dataset: {dataset_file}")

    # # MDP (only used to extract data shapes)
    # import quanser_robots
    # mdp = Gym(env_id)
    # dim_in = mdp.info.observation_space.shape[0]
    # dim_out = mdp.info.action_space.shape[0]
    dim_in = 1
    dim_out = 1

    train_dataset = toy_datasets.Sine1dDataset(
        data_spec=[(-0.75, -0.5, 100), (0, 0.25, 100), (0.75, 1, 100)],
    )
    test_dataset = toy_datasets.Sine1dDataset(
        data_spec=[(-3, 3, 500)],
    )

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

    # plotting data
    x_train, y_train = train_dataset[:]
    x_test, y_test = test_dataset[:]

    # # sort test data
    # xy_test = torch.cat([x_test, y_test], dim=1)
    # xy_test, _ = torch.sort(xy_test, dim=0)
    # x_test, y_test = xy_test.chunk(2, dim=1)

    fig, ax = plt.subplots(1, 1)

    ax.scatter(x_test, y_test, c="k", s=5)
    ax.plot(x_train, y_train, "mx", markersize=5)
    ax.set_ylabel("$y$")

    ax.set_ylabel("$y$")

    name = "snngp"
    ax.set_title(name)

    fig_trace, ax_trace = plt.subplots()
    ax_trace.plot(trace)
    ax_trace.set_title(name)
    mu_test, sigma_test, _, _ = model(x_test)

    plot_gp(ax, x_test, mu_test, torch.diag(sigma_test))

    f = model.features(x_test).detach()
    fig_features, ax_features = plt.subplots(figsize=(12, 9))
    ax_features.plot(x_test, f[:, ::10], alpha=0.25)
    ax_features.set_title(name)

    plt.show()
    # logger.finish()

    print(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    # print(f"Logs in {results_dir}")


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
