import math

import torch
import torch.nn as nn
from check_shape import check_shape
from torch.distributions import MultivariateNormal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.feature_fns.nns import TwoLayerNetwork, TwoLayerNormalizedResidualNetwork
from src.utils.conversion_utils import vec
from src.utils.whitening import data_whitening


class LinearBayesianModel(nn.Module):

    features: nn.Module

    def __init__(self, dim_x, dim_y, dim_features):
        super().__init__()

        # constants (saved)
        self.register_buffer("dim_x", torch.tensor(dim_x))
        self.register_buffer("dim_y", torch.tensor(dim_y))
        self.register_buffer("dim_features", torch.tensor(dim_features))
        self.register_buffer("prior_mean", torch.zeros(dim_features, dim_y))
        self.register_buffer("prior_var_in", torch.tensor(1 + 1e-3))
        self.register_buffer(
            "prior_cov_in", self.prior_var_in * torch.eye(dim_features)
        )
        self.register_buffer(
            "prior_cov_in_tril", torch.sqrt(self.prior_var_in) * torch.eye(dim_features)
        )

        # parameters (saved)
        self.post_mean = nn.Parameter(torch.zeros(dim_features, dim_y))
        ## parameterize in square root form to ensure positive definiteness
        self.post_cov_in_chol = nn.Parameter(torch.eye(dim_features))
        ## parameterize in square root form to ensure positive definiteness
        self.error_vars_out_sqrt = nn.Parameter(1e-2 * torch.ones(dim_y))
        # assume: error_cov_out = prior_cov_out = post_cov_out

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

    def error_cov_tril(self):
        return torch.diag(self.error_vars_out_sqrt)

    def sample_function(self):
        raise NotImplementedError

    def forward(self, x, grad=False, covs=True):
        """
        Predicts on x

        Parameter and return shapes see below.
        """
        n = x.shape[0]
        with torch.set_grad_enabled(grad):
            phi = self.features(x)
            mu = phi @ self.post_mean
            if covs:
                covariance_out = self.error_cov()
                covariance_feat = phi @ self.post_cov_in() @ phi.t()
                covariance_pred_in = torch.eye(n, device=x.device) + covariance_feat
                covariance = torch.kron(
                    covariance_out, covariance_pred_in
                )  # only useful for 1D?

        if covs:
            return mu, covariance, covariance_feat, covariance_out
        else:
            return mu

    def ellh(self, x, y):
        """
        Computes expected log likelihood of params given data x and y

        Parameter and return shapes see below.
        """
        with torch.set_grad_enabled(True):
            n = x.shape[0]
            phi = self.features(x)
            const = -n * self.dim_y * math.log(2 * math.pi) / 2
            w_ent = -n * self.error_cov().logdet() / 2
            y_pred = phi @ self.post_mean
            err = -0.5 * torch.trace(
                self.error_cov().inverse()
                * (y.T @ y - 2 * y.T @ y_pred + y_pred.T @ y_pred)
            )
            prec = -0.5 * torch.trace(self.post_cov_in() @ phi.T @ phi)
            llh = const + w_ent + err + prec
            return llh

    def kl(self):
        """
        Computes kl divergence of parameter posterior (approx) from
        parameter prior.

        Parameter and return shapes see below.
        """
        # assumption
        prior_cov_out = self.error_cov()
        prior_cov_out_tril = self.error_cov_tril()
        post_cov_out_tril = self.error_cov_tril()

        # use tril, because more efficient and numerically more stable
        if self.post_mean.shape[-1] == 1:
            div = kl_divergence(
                MultivariateNormal(
                    self.post_mean.t(), scale_tril=self.post_cov_in_tril()
                ),
                MultivariateNormal(
                    self.prior_mean.t(), scale_tril=self.prior_cov_in_tril
                ),
            )
            ### matrix kl
            div2 = (
                0.5
                * self.dim_y
                * torch.trace(self.prior_cov_in.inverse() @ self.post_cov_in())
                + 0.5
                * torch.trace(
                    (self.prior_mean - self.post_mean).T
                    @ self.prior_cov_in.inverse()
                    @ (self.prior_mean - self.post_mean)
                    @ prior_cov_out.inverse()
                )
                - 0.5
                * self.dim_y
                * (
                    self.dim_features
                    - self.prior_cov_in.logdet()
                    + self.post_cov_in().logdet()
                )
            )
            ### mvn kl (as in latex)
            div3 = 0.5 * (
                (self.prior_mean - self.post_mean).T
                @ self.prior_cov_in.inverse()
                @ (self.prior_mean - self.post_mean)
                + torch.trace(self.prior_cov_in.inverse() @ self.post_cov_in())
                - self.post_cov_in().logdet()
                + self.prior_cov_in.logdet()
                - self.dim_y * self.dim_features
            )
            ### mvn kl (as by torch)
            # half_term1 = (self.prior_cov_in_tril.diag().log().sum()
            #               - self.post_cov_in_tril().diag().log().sum())
            # term2 = torch.trace(
            #     torch.linalg.solve(self.prior_cov_in, self.post_cov_in()))
            half_term1 = (
                self.prior_cov_in_tril.logdet() - self.post_cov_in_tril().logdet()
            )
            term2 = torch.trace(self.prior_cov_in.inverse() @ self.post_cov_in())
            term3 = (
                (self.prior_mean - self.post_mean).T
                @ (self.prior_cov_in_tril @ self.prior_cov_in_tril.T).inverse()
                @ (self.prior_mean - self.post_mean)
            )
            div4 = half_term1 + 0.5 * (term2 + term3 - self.dim_features)
        else:
            div = kl_divergence(  # kron of tril ok, because cov_out diagonal!
                MultivariateNormal(
                    vec(self.post_mean),
                    scale_tril=torch.kron(post_cov_out_tril, self.post_cov_in_tril()),
                ),
                MultivariateNormal(
                    vec(self.prior_mean),
                    scale_tril=torch.kron(prior_cov_out_tril, self.prior_cov_in_tril),
                ),
            )
            ### matrix kl
            div2 = (
                0.5
                * self.dim_y
                * torch.trace(self.prior_cov_in.inverse() @ self.post_cov_in())
                + 0.5
                * torch.trace(
                    (self.prior_mean - self.post_mean).T
                    @ self.prior_cov_in.inverse()
                    @ (self.prior_mean - self.post_mean)
                    @ prior_cov_out.inverse()
                )
                - 0.5
                * self.dim_y
                * (
                    self.dim_features
                    - self.prior_cov_in.logdet()
                    + self.post_cov_in().logdet()
                )
            )
        return div

    def elbo(self, x, y):
        """
        should be equivalent to `ellh(x, y) - kl()`
        - is matrix normal compatible

        computes elbo in 4th form, i.e. L = E_q[log p(y|w)] - KL[q(w)||p(w|y)],
        expected log likelihood with w from variational posterior q(w) minus
        KL divergence of the variational posterior from the prior
        """
        # assumption
        prior_cov_out = self.error_cov()
        post_cov_out = self.error_cov()

        with torch.set_grad_enabled(True):
            n = x.shape[0]
            check_shape([x], [(n, self.dim_x.item())])
            check_shape([y], [(n, self.dim_y.item())])

            phi = self.features(x)
            # expected log likelihood
            part1a = -n / 2 * self.dim_y * math.log(2 * math.pi)
            part1b = -n / 2 * self.error_cov().logdet()
            y_pred = phi @ self.post_mean
            part2 = -0.5 * torch.trace(
                self.error_cov().inverse()
                * (y.T @ y - 2 * y.T @ y_pred + y_pred.T @ y_pred)
            )
            part3 = -0.5 * (
                torch.trace(self.error_cov().inverse() @ post_cov_out)  # cancels
                * torch.trace(self.post_cov_in() @ phi.T @ phi)
            )
            # kl of posterior from prior (vec kl works too: check speed?)
            part4 = (
                0.5
                * (torch.trace(self.prior_cov_in.inverse() @ self.post_cov_in()))
                * (torch.trace(prior_cov_out.inverse() @ post_cov_out))  # cancels
            )
            part5 = 0.5 * torch.trace(
                (self.prior_mean - self.post_mean).T
                @ self.prior_cov_in.inverse()
                @ (self.prior_mean - self.post_mean)
                @ post_cov_out.inverse()
            )
            part6 = -0.5 * (
                self.dim_features * self.dim_y
                + self.dim_y * self.post_cov_in().logdet()
                - self.dim_y * self.prior_cov_in.logdet()
                + self.dim_features * post_cov_out.logdet()  # cancels...
                - self.dim_features * prior_cov_out.logdet()  # ... with this
            )

            ellh = part1a + part1b + part2 + part3
            kl = part4 + part5 + part6
            return ellh - kl

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


@data_whitening
class SpectralNormalizedNeuralGaussianProcess(LinearBayesianModel):
    d_approx = 1024  # RFFs require ~512-1024 for accuracy

    def __init__(self, dim_x, dim_y, dim_features):
        super().__init__(dim_x, dim_y, self.d_approx)
        self.features = TwoLayerNormalizedResidualNetwork(
            dim_x, self.d_approx, dim_features
        )


@data_whitening
class NeuralLinearModel(LinearBayesianModel):
    def __init__(self, dim_x, dim_y, dim_features):
        super().__init__(dim_x, dim_y, dim_features)
        self.features = TwoLayerNetwork(dim_x, dim_features, dim_features)
