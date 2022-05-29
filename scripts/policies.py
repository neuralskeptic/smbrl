import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.parametrizations import spectral_norm
from torch.distributions import kl_divergence, MultivariateNormal
import math
from tqdm import tqdm


ACTIVATIONS = {
    "leakyrelu": functional.leaky_relu
}

import numpy as np  # temporary

# TODO: pytorch?
EPS = np.finfo(np.float64).tiny

def vec(x):
    """
    TODO pytorch
    https://stackoverflow.com/questions/25248290/most-elegant-implementation-of-matlabs-vec-function-in-numpy
    """
    shape = x.shape
    if len(shape) == 3:
        a, b, c = shape
        return x.reshape((a, b * c), order="F")
    else:
        return x.reshape((-1, 1), order="F")


def matrix_gaussian_kl(mean_1, cov_in_1, cov_out_1, mean_2, cov_in_2, cov_out_2):
    """
    TODO pytorch
    https://statproofbook.github.io/P/matn-kl
    """
    n, p = mean_1.shape
    diff = mean_2 - mean_1
    # MaVN covariances are scale invariant -- so we can safely normalize to prevent numerical issues.
    sf1 = p / np.trace(cov_out_1)
    sf2 = p / np.trace(cov_out_2)
    cov_out_1 = cov_out_1 * sf1
    cov_out_2 = cov_out_2 * sf2
    cov_in_1 = cov_in_1 / sf1
    cov_in_2 = cov_in_2 / sf2
    return (
        0.5
        * (
            n * np.log(max(EPS, np.linalg.det(cov_out_2)))
            - n * np.log(max(EPS, np.linalg.det(cov_out_1)))
            + p * np.log(max(EPS, np.linalg.det(cov_in_2)))
            - p * np.log(max(EPS, np.linalg.det(cov_in_1)))
            + np.trace(
                np.kron(
                    np.linalg.solve(cov_out_2, cov_out_1),
                    np.linalg.solve(cov_in_2, cov_in_1),
                )
            )
            + vec(diff).T
            @ vec(np.linalg.solve(cov_in_2, np.linalg.solve(cov_out_2, diff.T).T))
            - n * p
        ).item()
    )


def to_torch(x):
    return torch.from_numpy(x).float()


def autograd_tensor(x):
    """Same as torch.Tensor(x, requires_grad=True), but does not cause warnings."""
    return x.clone().detach().requires_grad_(True)


class Base(object):

    features: nn.Module

    def __init__(self, dim_x, dim_y, dim_features):
        self.dim_x, self.dim_y, self.d_features = dim_x, dim_y, dim_features
        self.mu_w = autograd_tensor(torch.zeros((dim_features, dim_y)))
        # parameterize in square root form to ensure positive definiteness
        self.sigma_w_chol = autograd_tensor(torch.eye(dim_features))
        # parameterize in square root form to ensure positive definiteness
        self.sigma_sqrt = autograd_tensor(1e-2 * torch.ones((dim_y,)))
        self.mu_w_prior = autograd_tensor(torch.zeros((dim_features, dim_y)))
        # parameterize in square root form to ensure positive definiteness
        self.sigma_w_prior_chol = autograd_tensor(torch.eye(dim_features))
        # parameterize in square root form to ensure positive definiteness
        self.sigma_prior_sqrt = autograd_tensor(torch.ones((dim_y,)))
        assert isinstance(self.features, nn.Module), "Features should have been specified by now."
        self.params = [self.mu_w, self.sigma_w_chol, self.sigma_sqrt] + list(
            self.features.parameters()
        )

    def sigma_w(self):
        return self.sigma_w_chol @ self.sigma_w_chol.t()

    def sigma_w_prior(self):
        # TODO make regularization more principled
        return self.sigma_w_prior_chol @ self.sigma_w_prior_chol.t() + 1e-3 * torch.eye(self.d_features)

    def sigma(self):
        return torch.diag(torch.pow(self.sigma_sqrt, 2))

    def sample_function(self):
        raise NotImplementedError

    def __call__(self, x):
        n = x.shape[0]
        with torch.set_grad_enabled(False):
            phi = self.features(x)

            mu = phi @ self.mu_w
            covariance_out = self.sigma()
            covariance_feat = phi @ self.sigma_w() @ phi.t()
            covariance_pred_in = torch.eye(n) + covariance_feat
            covariance = torch.kron(
                covariance_out, covariance_pred_in
            )  # only useful for 1D?
        return mu, covariance, covariance_feat, covariance_out

    def ellh(self, x, y):
        with torch.set_grad_enabled(True):
            n = x.shape[0]
            phi = self.features(x)
            const = -torch.tensor(n * math.log(2 * math.pi) / 2)
            w_ent = -n * self.sigma().logdet() / 2
            y_pred = phi @ self.mu_w
            err = -0.5 * torch.trace(
                self.sigma().inverse()
                * (y.t() @ y - 2 * y.t() @ y_pred + y_pred.t() @ y_pred)
            )
            prec = -0.5 * torch.trace(self.sigma_w() @ phi.t() @ phi)
            return const + w_ent + err + prec

    def kl(self):
        # TODO replace with matrix normal KL
        return kl_divergence(
            MultivariateNormal(self.mu_w.t(), self.sigma_w()),
            MultivariateNormal(self.mu_w_prior.t(), self.sigma_w_prior()))

    def train(self, x, y, n_iter=1000, lr=1e-3):
        # x, y = map(to_torch, [x, y])
        n_batch = x.shape[0]
        opt = torch.optim.Adam(self.params, lr=lr)
        trace = []
        for i in tqdm(range(n_iter)):
            opt.zero_grad()
            ellh = self.ellh(x, y)
            kl = self.kl()
            loss = -ellh + kl
            loss.backward()
            opt.step()
            trace.append(loss.detach().item())
        return trace


class TwoLayerNormalizedResidualNetwork(nn.Module):

    def __init__(self, d_in, d_out, d_hidden, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        # Note: spectral_norm sets norm to 1, do we need it to be configurable?
        self.l1 = spectral_norm(nn.Linear(d_in, d_hidden))
        self.l2 = spectral_norm(nn.Linear(d_hidden, d_hidden))
        self.act = ACTIVATIONS[activation]
        self.W = torch.randn(d_hidden, d_out // 2)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = l1 + self.act(l1)
        l3 = l2 + self.act(l2)
        return torch.cat([torch.cos(l3 @ self.W), torch.sin(l3 @ self.W)], 1)


class SpectralNormalizedNeuralGaussianProcess(Base):

    d_approx = 1024  # RFFs require ~512-1024 for accuracy

    def __init__(self,  dim_x, dim_y, dim_features):
        self.features = TwoLayerNormalizedResidualNetwork(dim_x, self.d_approx, dim_features)
        super().__init__(dim_x, dim_y, self.d_approx)


class TwoLayerNetwork(nn.Module):

    def __init__(self, d_in, d_out, d_hidden, activation="leakyrelu"):
        assert activation in ACTIVATIONS
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_out)
        self.act = ACTIVATIONS[activation]

    def forward(self, x):
        return self.act(self.l2(self.act(self.l1(x))))


class NeuralLinearModel(Base):

    def __init__(self,  dim_x, dim_y, dim_features):
        self.features = TwoLayerNetwork(dim_x, dim_features, dim_features)
        super().__init__(dim_x, dim_y, dim_features)


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
        trace = model.train(to_torch(x_train), to_torch(y_train), n_iter=100, lr=1e-3)

        fig_trace, ax_trace = plt.subplots()
        ax_trace.plot(trace)
        ax_trace.set_title(name)
        mu_test, sigma_test, _, _ = model(to_torch(x_test))

        plot_gp(ax, x_test, mu_test.numpy(), np.diag(sigma_test.numpy()))

        f = model.features(to_torch(x_test)).detach().numpy()
        fig_features, ax_features = plt.subplots(figsize=(12, 9))
        ax_features.plot(x_test, f[:, ::10], alpha=0.25)
        ax_features.set_title(name)

    plt.show()
