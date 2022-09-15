import math

import torch
import torch.nn as nn
from check_shape import check_shape
from torch.distributions import MultivariateNormal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.feature_fns.nns import TwoLayerNetwork, TwoLayerNormalizedResidualNetwork
from src.utils.conversion_utils import autograd_tensor


class LinearBayesianModel(object):

    features: nn.Module

    def __init__(self, dim_x, dim_y, dim_features, lr, device="cpu"):
        self.dim_x, self.dim_y, self.d_features = dim_x, dim_y, dim_features
        # self.mu_w = autograd_tensor(torch.zeros((dim_features, dim_y)))
        # self.mu_w = torch.zeros((dim_features, dim_y), device='cuda', requires_grad=True)
        self.mu_w = autograd_tensor(torch.zeros((dim_features, dim_y), device=device))
        # parameterize in square root form to ensure positive definiteness
        self.sigma_w_chol = autograd_tensor(torch.eye(dim_features, device=device))
        # parameterize in square root form to ensure positive definiteness
        self.sigma_sqrt = autograd_tensor(1e-2 * torch.ones((dim_y,), device=device))
        self.mu_w_prior = autograd_tensor(
            torch.zeros((dim_features, dim_y), device=device)
        )
        # parameterize in square root form to ensure positive definiteness
        self.sigma_w_prior_chol = autograd_tensor(
            # TODO make regularization more principled
            (1 + 1e-3)
            * torch.eye(dim_features, device=device)
        )
        # parameterize in square root form to ensure positive definiteness
        self.sigma_prior_sqrt = autograd_tensor(torch.ones((dim_y,), device=device))
        assert isinstance(
            self.features, nn.Module
        ), "Features should have been specified by now."
        self.params = [self.mu_w, self.sigma_w_chol, self.sigma_sqrt] + list(
            self.features.parameters()
        )

        self.lr = lr
        self.opt = torch.optim.Adam(self.params, lr=self.lr)

        self.device = device

    def state_dict(self):
        state_dict = {
            "mu_w": self.mu_w,
            "sigma_w_chol": self.sigma_w_chol,
            "sigma_sqrt": self.sigma_sqrt,
        }
        state_dict.update(self.features.state_dict())
        return state_dict

    def load_state_dict(self, state_dict):
        self.mu_w = state_dict.pop("mu_w")
        self.sigma_w_chol = state_dict.pop("sigma_w_chol")
        self.sigma_sqrt = state_dict.pop("sigma_sqrt")
        self.features.load_state_dict(state_dict)

    def sigma_w(self):
        lower_triangular = self.sigma_w_tril()
        return lower_triangular @ lower_triangular.t()

    def sigma_w_tril(self):
        # make diagonal of sigma_w_chol positive (softplus)
        # (this makes sure L@L.T is positive definite)
        diagonal = torch.diag_embed(
            torch.nn.functional.softplus(self.sigma_w_chol.diag())
        )
        lower_triangular_wo_diag = torch.tril(self.sigma_w_chol, diagonal=-1)
        lower_triangular = lower_triangular_wo_diag + diagonal
        return lower_triangular

    def sigma(self):
        return torch.diag(torch.pow(self.sigma_sqrt, 2))

    def sample_function(self):
        raise NotImplementedError

    def __call__(self, x):
        """
        Predicts on x

        Parameter and return shapes see below.
        """
        n = x.shape[0]
        with torch.set_grad_enabled(False):
            phi = self.features(x)
            mu = phi @ self.mu_w
            covariance_out = self.sigma()
            covariance_feat = phi @ self.sigma_w() @ phi.t()
            covariance_pred_in = torch.eye(n, device=self.device) + covariance_feat
            covariance = torch.kron(
                covariance_out, covariance_pred_in
            )  # only useful for 1D?

        check_shape([x], [(n, self.dim_x)])
        check_shape([mu], [(n, self.dim_y)])
        check_shape([covariance], [(n, n)])  # TODO dy1 update
        check_shape([covariance_feat], [(n, n)])
        check_shape([covariance_out], [(self.dim_y, self.dim_y)])
        return mu, covariance, covariance_feat, covariance_out

    def ellh(self, x, y):
        """
        Computes expected log likelihood of params given data x and y

        Parameter and return shapes see below.
        """
        with torch.set_grad_enabled(True):
            n = x.shape[0]
            phi = self.features(x)
            const = -torch.tensor(n * math.log(2 * math.pi) / 2, device=self.device)
            w_ent = -n * self.sigma().logdet() / 2
            y_pred = phi @ self.mu_w
            err = -0.5 * torch.trace(
                self.sigma().inverse()
                * (y.t() @ y - 2 * y.t() @ y_pred + y_pred.t() @ y_pred)
            )
            prec = -0.5 * torch.trace(self.sigma_w() @ phi.t() @ phi)
            llh = const + w_ent + err + prec

            check_shape([x], [(n, self.dim_x)])
            check_shape([y], [(n, self.dim_y)])
            check_shape([llh], [()])
            return llh

    def kl(self):
        """
        Computes kl divergence between parameter posterior (approx) and
        parameter prior.

        Parameter and return shapes see below.
        """
        # TODO replace with matrix normal KL
        # use tril, because more efficient and numerically more stable
        sigma_w_tril = self.sigma_w_tril()
        div = kl_divergence(
            MultivariateNormal(self.mu_w.t(), scale_tril=self.sigma_w_tril()),
            MultivariateNormal(self.mu_w_prior.t(), scale_tril=self.sigma_w_prior_chol),
        )
        check_shape([div], [(1,)])
        return div

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
                # loss = -ELBO in 3rd form (see https://en.wikipedia.org/wiki/Evidence_lower_bound#Main_forms)
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

    def __init__(self, dim_x, dim_y, dim_features, lr, device="cpu"):
        self.features = TwoLayerNormalizedResidualNetwork(
            dim_x, self.d_approx, dim_features, device=device
        )
        self.features.to(device)
        super().__init__(dim_x, dim_y, self.d_approx, lr, device=device)


class NeuralLinearModel(LinearBayesianModel):
    def __init__(self, dim_x, dim_y, dim_features, lr, device="cpu"):
        self.features = TwoLayerNetwork(dim_x, dim_features, dim_features)
        self.features.to(device)
        super().__init__(dim_x, dim_y, dim_features, lr, device=device)
