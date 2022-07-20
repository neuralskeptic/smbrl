import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.feature_fns.nns import TwoLayerNetwork, TwoLayerNormalizedResidualNetwork
from src.utils.conversion_utils import autograd_tensor


class LinearBayesianModel(object):

    features: nn.Module

    def __init__(self, dim_x, dim_y, dim_features, lr):
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
        assert isinstance(
            self.features, nn.Module
        ), "Features should have been specified by now."
        self.params = [self.mu_w, self.sigma_w_chol, self.sigma_sqrt] + list(
            self.features.parameters()
        )

        self.lr = lr
        self.opt = torch.optim.Adam(self.params, lr=self.lr)

    def sigma_w(self):
        lower_triangular = torch.tril(self.sigma_w_chol)
        return lower_triangular @ lower_triangular.t()

    def sigma_w_prior(self):
        # TODO make regularization more principled
        return self.sigma_w_prior_chol @ self.sigma_w_prior_chol.t() + 1e-3 * torch.eye(
            self.d_features
        )

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
            MultivariateNormal(self.mu_w_prior.t(), self.sigma_w_prior()),
        )

    def train(self, dataloader: DataLoader, n_epochs):
        trace = []
        for epoch in tqdm(range(n_epochs), position=0):
            for i_minibatch, minibatch in enumerate(
                tqdm(dataloader, leave=False, position=1)
            ):
                x, y = minibatch
                self.opt.zero_grad()
                ellh = self.ellh(x, y)
                kl = self.kl()
                loss = -ellh + kl
                loss.backward()
                self.opt.step()
                trace.append(loss.detach().item())
        return trace

    def train2(self, x, y, n_epochs):
        trace = []
        for epoch in tqdm(range(n_epochs)):
            self.opt.zero_grad()
            ellh = self.ellh(x, y)
            kl = self.kl()
            loss = -ellh + kl
            loss.backward()
            self.opt.step()
            trace.append(loss.detach().item())
        return trace


class SpectralNormalizedNeuralGaussianProcess(LinearBayesianModel):

    d_approx = 1024  # RFFs require ~512-1024 for accuracy

    def __init__(self, dim_x, dim_y, dim_features, lr):
        self.features = TwoLayerNormalizedResidualNetwork(
            dim_x, self.d_approx, dim_features
        )
        super().__init__(dim_x, dim_y, self.d_approx, lr)


class NeuralLinearModel(LinearBayesianModel):
    def __init__(self, dim_x, dim_y, dim_features, lr):
        self.features = TwoLayerNetwork(dim_x, dim_features, dim_features)
        super().__init__(dim_x, dim_y, dim_features, lr)
