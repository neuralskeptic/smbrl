import torch

from library import utils
from library.models.linear_bayesian_models import (
    NeuralLinearModel,
    SpectralNormalizedNeuralGaussianProcess,
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def test_function(x):
        return torch.sin(x * torch.pi)

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

    fig, ax = plt.subplots(1, 1)

    x_max = 1
    N = 1000

    x_train = torch.concat(
        (
            torch.distributions.Uniform(-1, 1).sample((N, 1)),
            # torch.distributions.Uniform(-1,1).sample([1,]),
        ),
        axis=0,
    )

    x_test = torch.linspace(-3, 3, 500)[:, None]
    y_train = test_function(x_train)
    y_test = test_function(x_test)

    # # whiten data here for simplicity
    # x_mean, x_std = x_train.mean(), x_train.std()
    # y_mean, y_std = y_train.mean(), y_train.std()
    # x_train = (x_train - x_mean) / x_std
    # x_test = (x_test - x_mean) / x_std
    # y_train = (y_train) / y_std
    # y_test = (y_test) / y_std

    ax.plot(x_test, y_test, "k-")
    ax.plot(x_train, y_train, "mx", markersize=5)
    ax.set_ylabel("$y$")

    ax.set_ylabel("$y$")

    name = "snngp"
    ax.set_title(name)
    model = SpectralNormalizedNeuralGaussianProcess(1, 1, 512)
    trace = model.train(x_train, y_train, n_iter=100, lr=4e-3)

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
