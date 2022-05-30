import torch

from library import utils
from library.models.linear_bayesian_models import NeuralLinearModel, SpectralNormalizedNeuralGaussianProcess

import numpy as np  # temporary

# TODO: numpy -> pytorch?
EPS = np.finfo(np.float64).tiny

if __name__ == "__main__":
    """
    TODO:
    1. multivariate regression
    2. how well do we fit functions compared to standard MLPs and GPs
    3. How well do shared features perform?
    4. is training stable w.r.t. pos-def covar
    5. why does GPy not work for me?
    """
    import matplotlib.pyplot as plt

    def test_function(x):
        # return np.sin(x)
        return 1. * (np.sin(x) > 0.)

    def plot_gp(axis, x, mu, var):
        axis.plot(x, mu, "b-")
        for n_std in range(1, 3):
            std = n_std * np.sqrt(var.squeeze())
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

    x_max = 6 * np.pi

    fig, axs = plt.subplots(3, 1, figsize=(10, 21), sharex=True, sharey=True)

    # x_train = np.array(
    #     [
    #         -5.8,
    #         -3.1,
    #         -2.8,
    #         -2.75,
    #         2.71,
    #         2.1,
    #         6.5,
    #         6.6,
    #     ]
    # )[:, None]
    x_train = np.concatenate((
        np.random.uniform(low=-3*np.pi, high=-np.pi, size=(25,1)),
        np.random.uniform(low=1*np.pi, high=3 * np.pi, size=(25,1))
        ),
        axis=0)
    print()
    x_test = np.linspace(-x_max, x_max, 500)[:, None]
    y_train = test_function(x_train)
    y_test = test_function(x_test)

    # whiten data here for simplicity
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train = (y_train) / y_std
    y_test = (y_test) / y_std

    for ax in axs.flatten():
        ax.plot(x_test, y_test, "k-")
        ax.plot(x_train, y_train, "mx", markersize=5)

    for ax in axs:
        ax.set_ylabel("$y$")

    axs[-1].set_ylabel("$y$")

    import GPy
    # kern = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.) + GPy.kern.White(1, variance=1e-6)
    m = GPy.models.GPRegression(x_train, y_train)
    # m.likelihood.variance = 1e-5
    # m.likelihood.variance.fix()
    m.randomize()
    m.optimize()
    print(m)
    mu_y, var_y = m.predict(x_test)
    plot_gp(axs[0], x_test, mu_y, var_y)
    axs[0].set_title("ngp (se)")

    models = {
        "nlm": NeuralLinearModel,
        "snngp": SpectralNormalizedNeuralGaussianProcess
    }
    for (name, model_class), ax in zip(models.items(), axs[1:]):
        ax.set_title(name)
        model = model_class(1, 1, 512)
        trace = model.train(utils.to_torch(x_train), utils.to_torch(y_train), n_iter=1, lr=1e-3)
        # trace = model.train(utils.to_torch(x_train), utils.to_torch(y_train), n_iter=100, lr=1e-3)

        fig_trace, ax_trace = plt.subplots()
        ax_trace.plot(trace)
        ax_trace.set_title(name)
        mu_test, sigma_test, _, _ = model(utils.to_torch(x_test))

        plot_gp(ax, x_test, mu_test.numpy(), np.diag(sigma_test.numpy()))

        f = model.features(utils.to_torch(x_test)).detach().numpy()
        fig_features, ax_features = plt.subplots(figsize=(12, 9))
        ax_features.plot(x_test, f[:, ::10], alpha=0.25)
        ax_features.set_title(name)

    plt.show()
