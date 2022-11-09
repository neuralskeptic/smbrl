import math

import torch
import torch.nn as nn
from check_shape import check_shape
from torch.distributions import MultivariateNormal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.feature_fns.nns import TwoLayerNetwork, TwoLayerNormalizedResidualNetwork
from src.utils.conversion_utils import autograd_tensor, vec


class LinearBayesianModel(object):

    features: nn.Module

    def __init__(self, dim_x, dim_y, dim_features, lr, device="cpu"):
        self.dim_x, self.dim_y, self.d_features = dim_x, dim_y, dim_features
        # self.post_mean = autograd_tensor(torch.zeros((dim_features, dim_y)))
        # self.post_mean = torch.zeros((dim_features, dim_y), device='cuda', requires_grad=True)
        self.post_mean = autograd_tensor(
            torch.zeros((dim_features, dim_y), device=device)
        )
        # parameterize in square root form to ensure positive definiteness
        self.post_cov_in_chol = autograd_tensor(torch.eye(dim_features, device=device))
        # parameterize in square root form to ensure positive definiteness
        self.error_vars_out_sqrt = autograd_tensor(
            1e-2 * torch.ones((dim_y,), device=device)
        )
        self.prior_mean = autograd_tensor(
            torch.zeros((dim_features, dim_y), device=device)
        )
        # parameterize in square root form to ensure positive definiteness
        self.prior_var_in = torch.tensor(1 + 1e-3).to(device=device)
        self.prior_cov_in_chol = self.prior_var_in * torch.eye(dim_features).to(device)
        # parameterize in square root form to ensure positive definiteness
        self.prior_cov_out = autograd_tensor(
            torch.ones((dim_y,), device=device)
        )  # TODO grad?
        assert isinstance(
            self.features, nn.Module
        ), "Features should have been specified by now."
        self.params = [
            self.post_mean,
            self.post_cov_in_chol,
            self.error_vars_out_sqrt,
        ] + list(self.features.parameters())

        self.lr = lr
        self.opt = torch.optim.Adam(self.params, lr=self.lr)

        self.device = device

        # whitening params (unused if no whitening)
        self.whitening = False
        self.x_mean = None
        self.w_PCA = None
        self.w_ZCA = None
        self.y_mean = None
        self.y_std = None

    def with_whitening(self, X, Y):
        # X, Y = X.to(self.device), Y.to(self.device)
        self.whitening = True
        # ZCA for X
        self.x_mean = X.mean(dim=0)  # along N
        # L, Q = torch.linalg.eigh(X.T.cov())  # eig of Cov = Q diag(L) Q'
        # self.w_PCA = torch.linalg.solve(torch.diag(L + 1e-5), Q.T)
        # self.w_ZCA = Q @ self.w_PCA

        x_centered = X - self.x_mean
        sigma = x_centered.t() @ x_centered / X.shape[0]
        U, S, _ = torch.svd(sigma)
        self.w_ZCA = U @ torch.diag(torch.sqrt(S + 1e-6) ** -1) @ U.t()

        self.x_mean = self.x_mean.to(self.device)
        self.w_ZCA = self.w_ZCA.to(self.device)

        # z-score for Y
        self.y_mean = Y.mean(dim=0).to(self.device)  # along N
        self.y_std = Y.std(dim=0).to(self.device)  # along N

    def whitenX(self, x):
        # return x  # uncomment to disable whitening
        # return (x - self.x_mean.detach()) @ self.w_PCA.detach()
        return (x - self.x_mean.detach()) @ self.w_ZCA.detach()

    def whitenY(self, y):
        return y  # uncomment to disable whitening
        # return (y - self.y_mean.detach()) / self.y_std.detach()

    def dewhitenY(self, y):
        return y  # uncomment to disable whitening
        # return y * self.y_std.detach() + self.y_mean.detach()

    def state_dict(self):
        state_dict = {
            "post_mean": self.post_mean,
            "post_cov_in_chol": self.post_cov_in_chol,
            "error_vars_out_sqrt": self.error_vars_out_sqrt,
            "whitening": self.whitening,
            "x_mean": self.x_mean,
            "w_PCA": self.w_PCA,
            "w_ZCA": self.w_ZCA,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }
        state_dict.update(self.features.state_dict())
        return state_dict

    def load_state_dict(self, state_dict):
        self.post_mean = state_dict.pop("post_mean")
        self.post_cov_in_chol = state_dict.pop("post_cov_in_chol")
        self.error_vars_out_sqrt = state_dict.pop("error_vars_out_sqrt")
        self.whitening = state_dict.pop("whitening")
        self.x_mean = state_dict.pop("x_mean")
        self.w_PCA = state_dict.pop("w_PCA")
        self.w_ZCA = state_dict.pop("w_ZCA")
        self.y_mean = state_dict.pop("y_mean")
        self.y_std = state_dict.pop("y_std")
        self.features.load_state_dict(state_dict)

    def prior_cov_in(self):
        return self.prior_cov_in_chol @ self.prior_cov_in_chol.T

    def post_cov_in(self):
        lower_triangular = self.post_cov_in_tril()
        return lower_triangular @ lower_triangular.t()

    def post_cov_in_tril(self):
        # make diagonal of post_cov_in_chol positive (softplus)
        # (this makes sure L@L.T is positive definite)
        diagonal = torch.diag_embed(
            torch.nn.functional.softplus(self.post_cov_in_chol.diag())
        )
        lower_triangular_wo_diag = torch.tril(self.post_cov_in_chol, diagonal=-1)
        lower_triangular = lower_triangular_wo_diag + diagonal
        return lower_triangular

    def error_cov(self):
        return torch.diag(torch.pow(self.error_vars_out_sqrt, 2))

    def sample_function(self):
        raise NotImplementedError

    def __call__(self, x, grad=False, covs=True):
        """
        Predicts on x

        Parameter and return shapes see below.
        """
        if self.whitening:
            x = self.whitenX(x)

        n = x.shape[0]
        with torch.set_grad_enabled(grad):
            phi = self.features(x)
            mu = phi @ self.post_mean
            if covs:
                covariance_out = self.error_cov()
                covariance_feat = phi @ self.post_cov_in() @ phi.t()
                covariance_pred_in = torch.eye(n, device=self.device) + covariance_feat
                covariance = torch.kron(
                    covariance_out, covariance_pred_in
                )  # only useful for 1D?

        if self.whitening:
            mu = self.dewhitenY(mu)
            # dewhiten covariance output?

        # check_shape([x], [(n, self.dim_x)])
        # check_shape([mu], [(n, self.dim_y)])
        # check_shape([covariance], [(n, n)])  # TODO dy1 update
        # check_shape([covariance_feat], [(n, n)])
        # check_shape([covariance_out], [(self.dim_y, self.dim_y)])
        if covs:
            return mu, covariance, covariance_feat, covariance_out
        else:
            return mu

    def ellh(self, x, y):
        """
        Computes expected log likelihood of params given data x and y

        Parameter and return shapes see below.
        """
        if self.whitening:
            x = self.whitenX(x)
            y = self.whitenY(y)

        with torch.set_grad_enabled(True):
            n = x.shape[0]
            phi = self.features(x)
            const = -torch.tensor(
                n * self.dim_y * math.log(2 * math.pi) / 2, device=self.device
            )
            w_ent = -n * self.error_cov().logdet() / 2
            y_pred = phi @ self.post_mean
            err = -0.5 * torch.trace(
                self.error_cov().inverse()
                * (y.T @ y - 2 * y.T @ y_pred + y_pred.T @ y_pred)
            )
            prec = -0.5 * torch.trace(phi @ self.post_cov_in() @ phi.T)
            llh = const + w_ent + err + prec

            check_shape([x], [(n, self.dim_x)])
            check_shape([y], [(n, self.dim_y)])
            check_shape([llh], [()])
            return llh

    def kl(self):
        """
        Computes kl divergence of parameter posterior (approx) from
        parameter prior.

        Parameter and return shapes see below.
        """
        # TODO replace with matrix normal KL
        # use tril, because more efficient and numerically more stable
        div = kl_divergence(
            MultivariateNormal(
                vec(self.post_mean.t()),
                covariance_matrix=torch.kron(
                    self.prior_cov_out, self.post_cov_in()  # = post cov out
                ),
            ),
            MultivariateNormal(
                vec(self.prior_mean.t()),
                covariance_matrix=torch.kron(self.prior_cov_out, self.prior_cov_in()),
            ),
        )
        check_shape([div], [(1,)])
        return div

    def elbo(self, x, y):
        """
        should be equivalent to `ellh(x, y) - kl()`
        - is matrix normal compatible

        computes elbo in 4th form, i.e. L = E_q[log p(y|w)] - KL[q(w)||p(w|y)],
        expected log likelihood with w from variational posterior q(w) minus
        KL divergence of the variational posterior from the prior
        """
        with torch.set_grad_enabled(True):
            n = x.shape[0]
            check_shape([x], [(n, self.dim_x)])
            check_shape([y], [(n, self.dim_y)])

            phi = self.features(x)
            # expected log likelihood
            part1 = -torch.tensor(n / 2 * self.dim_y * math.log(2 * math.pi))
            part1 = part1.to(self.device)
            part2 = -n / 2 * self.error_cov().logdet()  # manually with diag?
            y_pred = phi @ self.post_mean
            part3 = -0.5 * torch.trace(
                self.error_cov().inverse()
                * (y.T @ y - 2 * y.T @ y_pred + y_pred.T @ y_pred)
            )
            part4a = -1 / 2 * torch.trace(self.error_cov().inverse())
            part4b = torch.trace(self.post_cov_in() @ phi.T @ phi)
            # kl of posterior from prior
            part5 = (
                -0.5
                * self.dim_y
                / self.prior_var_in
                * (torch.trace(self.post_cov_in()))
            )
            part6 = (
                -0.5
                / self.prior_var_in
                * (torch.trace(self.post_mean.T @ self.post_mean))
            )
            part7 = 0.5 * (
                self.dim_x * self.dim_y
                + self.dim_x * self.dim_y * torch.log(1 / self.prior_var_in)
                - self.dim_y * self.post_cov_in().logdet()
            )

            elbo_value = part1 + part2 + part3 + part4a + part4b + part5 + part6 + part7
            return elbo_value

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
