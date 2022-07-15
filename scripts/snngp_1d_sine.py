import torch
from library import utils
from library.datasets import toy_datasets
from library.models.linear_bayesian_models import (
    SpectralNormalizedNeuralGaussianProcess,
)
from torch.utils.data import DataLoader

if __name__ == "__main__":
    SEED = 1234
    utils.set_random_seed(SEED)

    import matplotlib.pyplot as plt

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

    train_dataset = toy_datasets.Sine1dDataset(
        data_spec=[(-0.75, -0.5, 100), (0, 0.25, 100), (0.75, 1, 100)],
    )
    test_dataset = toy_datasets.Sine1dDataset(
        data_spec=[(-3, 3, 500)],
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # # whiten data here for simplicity
    # x_mean, x_std = x_train.mean(), x_train.std()
    # y_mean, y_std = y_train.mean(), y_train.std()
    # x_train = (x_train - x_mean) / x_std
    # x_test = (x_test - x_mean) / x_std
    # y_train = (y_train) / y_std
    # y_test = (y_test) / y_std

    model = SpectralNormalizedNeuralGaussianProcess(1, 1, 512)
    trace = model.train(train_dataloader, n_epochs=5, lr=4e-3)

    ### plotting
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
