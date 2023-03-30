import math
from dataclasses import dataclass
from typing import Callable

import einops
import torch
from torch.autograd.functional import hessian, jacobian

from src.i2c.distributions import MultivariateGaussian


class LinearizationInference(object):
    def __call__(self, function: Callable, distribution: MultivariateGaussian):
        function_ = lambda x: function(x)[0]
        jac = jacobian(function, distribution.mean)
        mean = function(distribution.mean[None, :])[0][0, :]
        covariance = jac @ distribution.covariance @ jac.T
        return MultivariateGaussian(
            mean,
            covariance,
            jac @ distribution.covariance,
            distribution.mean,
            distribution.covariance,
        )

    def mean(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> torch.float32:
        # TODO (joemwatson) only want to use this for R^d->R functions, but what about hessian usage generally?
        hess = hessian(function, distribution.mean[None, :])[0, :, 0, :]
        return function(distribution.mean[None, :]) + torch.trace(
            distribution.covariance @ hess
        )


class LinearizationInnovation(LinearizationInference):
    def __call__(
        self,
        function: Callable,
        inverse_temperature: float,
        distribution: MultivariateGaussian,
    ) -> MultivariateGaussian:
        jac = jacobian(function, distribution.mean[None, :])[0, 0, :]
        hess = hessian(function, distribution.mean[None, :])[0, :, 0, :]
        if (torch.linalg.eigvalsh(hess) < 0.0).any():
            hess = jac[:, None] @ jac[None, :]
        # import pdb; pdb.set_trace()
        precision = inverse_temperature * hess + torch.linalg.inv(
            distribution.covariance
        )
        covariance = torch.linalg.inv(precision)
        mean = distribution.mean - covariance @ (inverse_temperature * jac)
        if (torch.linalg.det(covariance) < 0.0).any():
            import pdb

            pdb.set_trace()
        return MultivariateGaussian(mean, covariance)


@dataclass
class CubatureQuadrature(object):
    """Implements sparse spherical cubature rule (\alpha=1) and the 'unscented' heuristic weights."""

    alpha: float
    beta: float
    kappa: float

    @staticmethod
    def pts(dim: int) -> torch.Tensor:
        return torch.cat(
            (torch.zeros((1, dim)), torch.eye(dim), -torch.eye(dim)), axis=0
        )

    def weights(self, dim):
        assert self.alpha > 0
        lam = self.alpha**2 * (dim + self.kappa) - dim
        sf = math.sqrt(dim + lam)
        w0_sig_extra = 1 - self.alpha**2 + self.beta
        weights_mu = (1 / (2 * (dim + lam))) * torch.ones((1 + 2 * dim,))
        weights_mu[0] = 2 * lam * weights_mu[0]
        weights_sig = weights_mu.clone()
        weights_sig[0] += w0_sig_extra
        print(weights_mu)
        return sf, weights_mu, weights_sig


@dataclass
class GaussHermiteQuadrature(object):
    degree: int

    def __post_init__(self):
        assert self.degree >= 1
        import numpy as np

        self.gh_pts, self.gh_weights = map(
            torch.from_numpy, np.polynomial.hermite.hermgauss(self.degree)
        )

    def pts(self, dim):
        grid = torch.meshgrid(*(self.gh_pts,) * dim, indexing="xy")
        return torch.vstack(tuple(map(torch.ravel, grid))).T

    def weights(self, dim):
        grid = torch.meshgrid(*(self.gh_weights,) * dim, indexing="xy")
        w = torch.vstack(tuple(map(torch.ravel, grid))).T
        w = torch.prod(w, axis=1) / (math.pi ** (dim / 2))
        print(w)
        return math.sqrt(2), w, w


class QuadratureInference(object):
    def __init__(self, dimension: int, params: CubatureQuadrature):
        self.dim = dimension
        self.base_pts = params.pts(dimension)
        self.sf, self.weights_mu, self.weights_sig = params.weights(dimension)
        self.n_points = self.base_pts.shape[0]

    def get_x_pts(self, mu_x, sigma_x):
        """ " batched version (batch dims first)"""
        try:
            sqrt = torch.linalg.cholesky(sigma_x)  # TODO full cuda version?
        except torch.linalg.LinAlgError:
            print(
                f"ERROR: singular covariance: {torch.linalg.eigvalsh(sigma_x)}, state: {mu_x}"
            )
            # TODO inject variance (good idea?)
            print(
                "ERROR: singular covariance: taking only diagonal and clipping to min 1e-8"
            )
            sigma_min = 1e-8
            diags = sigma_x.diagonal(dim1=-2, dim2=-1)
            diags = diags.maximum(torch.tensor(sigma_min))
            sqrt = torch.linalg.cholesky(diags.diag_embed())
        scale = self.sf * sqrt
        # mu + self.base_pts @ scale.T  <==>  (B,X) + (P,X) @ (B,X,X) = (B,P,X)
        return mu_x.unsqueeze(-2) + einops.einsum(
            self.base_pts, scale.mT, "p x2, ... x2 x1 -> ... p x1"
        )

    def __call__(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> MultivariateGaussian:
        """ " batched version (batch dims first)"""
        assert isinstance(distribution, MultivariateGaussian)
        x_pts = self.get_x_pts(distribution.mean, distribution.covariance)
        y_pts, m_y, sig_y, sig_xy = self.forward_pts(function, distribution.mean, x_pts)
        return MultivariateGaussian(
            m_y,
            sig_y,
            sig_xy.mT,  # transpose
            distribution.mean,
            distribution.covariance,
        )

    def forward_pts(
        self, f: Callable, mu_x: torch.Tensor, x_pts: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """ " batched version (batch dims first)"""
        y_pts, x_pts = f(x_pts)  # return updated arg, because env clips u
        mu_y = einops.einsum(self.weights_mu, y_pts, "p, ... p y -> ... y")
        diff_x = x_pts - einops.rearrange(mu_x, "... x -> ... 1 x")  # unsqueeze
        diff_y = y_pts - einops.rearrange(mu_y, "... y -> ... 1 y")
        dim_y = y_pts.shape[-1]
        sigma_y = einops.einsum(
            self.weights_sig,
            diff_y,
            diff_y,
            "p, ... p y1, ... p y2 -> ... y1 y2",
        ) + 1e-7 * torch.eye(
            dim_y, device=mu_x.device
        )  # regularization
        sigma_xy = einops.einsum(
            self.weights_sig, diff_x, diff_y, "p, ... p x, ... p y -> ... x y"
        )
        return y_pts, mu_y, sigma_y, sigma_xy

    def mean(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> torch.float32:
        return torch.einsum(
            "b,b->",
            self.weights_mu,
            function(self.get_x_pts(distribution.mean, distribution.covariance))[0],
        )

    def to(self, device):
        self.base_pts = self.base_pts.to(device)
        self.weights_mu = self.weights_mu.to(device)
        self.weights_sig = self.weights_sig.to(device)
        return self

    def cpu(self):
        return self.to("cpu")


class QuadratureGaussianInference(QuadratureInference):
    """For a function that returns a mean and covariance."""

    def forward_pts(
        self, f: Callable, mu_x: torch.Tensor, x_pts: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """ " batched version (batch dims first)"""
        # TODO get x_pts batch and outputs blockwise (!) sigmas
        y_pts, sigma_y_pts, x_pts = f(x_pts)
        mu_y = einops.einsum(self.weights_mu, y_pts, "p, ... p y -> ... y")
        diff_x = x_pts - einops.rearrange(mu_x, "... x -> ... 1 x")  # unsqueeze
        diff_y = y_pts - einops.rearrange(mu_y, "... y -> ... 1 y")
        # sigma_y = (y_pts - mu_y).T @ Wsig @ (y_pts - mu_y) + Wmu @ sigma_y_pts
        sigma_y = einops.einsum(
            self.weights_sig,
            diff_y,
            diff_y,
            "p, ... p y1, ... p y2 -> ... y1 y2",
        ) + einops.einsum(self.weights_mu, sigma_y_pts, "p, ... p y1 y2 -> ... y1 y2")
        sigma_xy = einops.einsum(
            self.weights_sig, diff_x, diff_y, "p, ... p x, ... p y -> ... x y"
        )
        return y_pts, mu_y, sigma_y, sigma_xy


class QuadratureImportanceSamplingInnovation(QuadratureInference):
    def __call__(
        self,
        function: Callable,
        inverse_temperature: float,
        distribution: MultivariateGaussian,
    ) -> MultivariateGaussian:
        """ " batched version (batch dims first)"""
        assert isinstance(distribution, MultivariateGaussian)
        x_pts = self.get_x_pts(distribution.mean, distribution.covariance)
        f_pts, x_pts = function(x_pts)
        log_weights = -inverse_temperature * f_pts + torch.log(self.weights_mu)
        log_weights -= torch.logsumexp(log_weights, dim=-1, keepdim=True)
        weights = torch.exp(log_weights)
        mu_x = einops.einsum(weights, x_pts, "... p, ... p x -> ... x")
        diff_x = x_pts - einops.rearrange(mu_x, "... x -> ... 1 x")  # unsqueeze
        sigma_x = einops.einsum(
            weights, diff_x, diff_x, "... p, ... p x1, ... p x2 -> ... x1 x2"
        )
        sigma_x_T = einops.rearrange(sigma_x, "... x1 x2 -> ... x2 x1")  # transpose
        sigma_x = 0.5 * (sigma_x + sigma_x_T)  # TODO why this?
        return MultivariateGaussian(mu_x, sigma_x)
