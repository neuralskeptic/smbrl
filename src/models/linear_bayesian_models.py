import math

import einops
import torch
import torch.nn as nn
from check_shape import check_shape

from src.feature_fns.nns import (
    MultiLayerPerceptron,
    ResidualNetwork,
    SpectralNormResidualNetwork,
)
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
        # caching
        self.register_buffer("prior_cov_in_inverse", self.prior_cov_in.inverse())
        self.register_buffer("prior_cov_in_logdet", self.prior_cov_in.logdet())

        # parameters (saved)
        self.post_mean = nn.Parameter(torch.zeros(dim_features, dim_y))
        ## parameterize in square root form to ensure positive definiteness
        self.post_cov_in_chol = nn.Parameter(torch.eye(dim_features))
        ## parameterize in square root form to ensure positive definiteness
        self.error_vars_out_sqrt = nn.Parameter(1e-2 * torch.ones(dim_y))
        ## parameterize in square root form to ensure positive definiteness
        self.pred_var_bias_sqrt = nn.Parameter(1e-2 * torch.ones(dim_y))

        # assumptions/simplifications
        # 1: LEARN (diag) error output covariance = post out cov = prior out cov
        # 2: LEARN (diag) error output covariance
        #    && LEARN (chol) post out cov = prior out cov
        # 3: LEARN (diag) error output covariance
        #    && LEARN (chol) post out cov
        #    && CONST (c*I) prior out cov
        ################## CHANGE ME ###################
        self.assumption = 1
        ################################################
        if self.assumption == 1:
            pass  # assume: error_cov_out = prior_cov_out = post_cov_out
        elif self.assumption == 2:
            self.post_cov_out_chol = nn.Parameter(torch.eye(dim_y))
        elif self.assumption == 3:
            self.post_cov_out_chol = nn.Parameter(torch.eye(dim_y))
            self.register_buffer("prior_var_out", torch.tensor(1 + 1e-3))
            self.register_buffer("prior_cov_out", self.prior_var_out * torch.eye(dim_y))

    # check eigenvalues of learnt cov matrices
    # plt.plot(torch.linalg.eigh(self.error_cov_out())[0].cpu().detach())
    # plt.plot(torch.linalg.eigh(self.post_cov_in())[0].cpu().detach())
    # plt.plot(torch.linalg.eigh(self.post_cov_out())[0].cpu().detach())

    def post_cov_in_tril(self):
        # make diagonal of post_cov_in_chol positive (softplus)
        # (this makes sure L@L.T is positive definite)
        diagonal = torch.diag_embed(
            torch.nn.functional.softplus(self.post_cov_in_chol.diag())
        )
        lower_triangular_wo_diag = torch.tril(self.post_cov_in_chol, diagonal=-1)
        lower_triangular = lower_triangular_wo_diag + diagonal
        return lower_triangular

    def post_cov_in(self):
        lower_triangular = self.post_cov_in_tril()
        return lower_triangular @ lower_triangular.mT

    def post_cov_out_tril(self):
        # make diagonal of post_cov_in_chol positive (softplus)
        # (this makes sure L@L.T is positive definite)
        diagonal = torch.diag_embed(
            torch.nn.functional.softplus(self.post_cov_out_chol.diag())
        )
        lower_triangular_wo_diag = torch.tril(self.post_cov_out_chol, diagonal=-1)
        lower_triangular = lower_triangular_wo_diag + diagonal
        return lower_triangular

    def post_cov_out(self):
        lower_triangular = self.post_cov_out_tril()
        return lower_triangular @ lower_triangular.mT

    def error_vars_out(self):
        return torch.pow(self.error_vars_out_sqrt, 2)

    def error_cov_out(self):
        return torch.diag(torch.pow(self.error_vars_out_sqrt, 2))

    def pred_var_bias(self):
        return torch.pow(self.pred_var_bias_sqrt, 2)

    def error_cov_out_tril(self):
        return torch.diag(self.error_vars_out_sqrt)

    def sample_function(self):
        raise NotImplementedError

    def forward(self, x):
        """
        Returns predictive posterior sampled on x

        Parameter and return shapes see below.
        """
        phi = self.features(x)
        mu = phi @ self.post_mean
        # only compute diagonal (batch cross-covariance not used)
        vars_feat = einops.einsum(
            phi, self.post_cov_in(), phi, "... x1, x1 x2, ... x2 -> ..."
        )
        vars_pred_in = vars_feat + 1
        cov_pred_out = self.error_cov_out()
        vars_pred = einops.einsum(vars_pred_in, cov_pred_out, "..., o1 o2 -> ... o1 o2")
        vars_pred += self.pred_var_bias()
        return mu, vars_pred

    def elbo(self, x, y):
        """
        should be equivalent to `ellh(x, y) - kl()`
        - is matrix normal compatible

        computes elbo in 3rd form
            L = E_q[log p(y|w)] - KL[q(w)||p(w)],
            (wikipedia: https://en.wikipedia.org/wiki/Evidence_lower_bound#Main_forms)
        expected log likelihood with w from variational posterior q(w) minus
        KL divergence of the variational posterior from the prior
        """
        # assumption
        if self.assumption == 1:
            prior_cov_out = self.error_cov_out()  # prior == error
            post_cov_out = self.error_cov_out()  # post == error
        elif self.assumption == 2:
            prior_cov_out = self.post_cov_out()  # prior == post
            post_cov_out = self.post_cov_out()  # learn
        elif self.assumption == 3:
            prior_cov_out = self.prior_cov_out  # const
            post_cov_out = self.post_cov_out()  # learn

        n = x.shape[0]
        check_shape([x], [(n, self.dim_x.item())])
        check_shape([y], [(n, self.dim_y.item())])

        phi = self.features(x)
        # expected log likelihood
        part1a = -n / 2 * self.dim_y * math.log(2 * math.pi)
        part1b = -n / 2 * self.error_cov_out().logdet()
        y_pred = phi @ self.post_mean
        part2 = -0.5 * torch.trace(
            (1 / self.error_vars_out()).diag()
            * (y.T @ y - 2 * y.T @ y_pred + y_pred.T @ y_pred)
        )
        part3 = -0.5 * (
            torch.trace((1 / self.error_vars_out()).diag() @ post_cov_out)  # = dim_y
            * torch.trace(self.post_cov_in() @ phi.T @ phi)
        )
        # kl of posterior from prior (vec kl works too: check speed?)
        part4 = (
            0.5
            * torch.trace(self.prior_cov_in_inverse @ self.post_cov_in())
            * torch.trace(prior_cov_out.inverse() @ post_cov_out)  # = dim_y
        )
        part5 = 0.5 * torch.trace(
            (self.prior_mean - self.post_mean).T
            @ self.prior_cov_in_inverse
            @ (self.prior_mean - self.post_mean)
            @ post_cov_out.inverse()
        )
        part6 = -0.5 * (
            self.dim_features * self.dim_y
            + self.dim_y * self.post_cov_in().logdet()
            - self.dim_y * self.prior_cov_in_logdet
            # + self.dim_features * post_cov_out.logdet()  # cancels...
            # - self.dim_features * prior_cov_out.logdet()  # ... with this
        )

        ellh = part1a + part1b + part2 + part3
        kl = part4 + part5 + part6
        return ellh - kl


@data_whitening
class SpectralNormalizedNeuralGaussianProcess(LinearBayesianModel):
    def __init__(self, dim_x, n_hidden_layers, dim_hidden, dim_features, dim_y):
        super().__init__(dim_x, dim_y, dim_features)
        self.features = SpectralNormResidualNetwork(
            dim_x,
            n_hidden_layers,
            dim_hidden,
            dim_features,
        )


@data_whitening
class NeuralLinearModelMLP(LinearBayesianModel):
    def __init__(self, dim_x, n_hidden_layers, dim_hidden, dim_features, dim_y):
        super().__init__(dim_x, dim_y, dim_features)
        self.features = MultiLayerPerceptron(
            dim_x, n_hidden_layers, dim_hidden, dim_features
        )


@data_whitening
class NeuralLinearModelResNet(LinearBayesianModel):
    def __init__(self, dim_x, n_hidden_layers, dim_hidden, dim_features, dim_y):
        super().__init__(dim_x, dim_y, dim_features)
        self.features = ResidualNetwork(
            dim_x, n_hidden_layers, dim_hidden, dim_features
        )
