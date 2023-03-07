import functools
import math
import os
import time
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from experiment_launcher import run_experiment
from experiment_launcher.utils import save_args
from mushroom_rl.core.logger.logger import Logger
from pytorch_minimize.optim import MinimizeWrapper
from torch.autograd.functional import hessian, jacobian
from tqdm import trange

from src.datasets.mutable_buffer_datasets import ReplayBuffer
from src.feature_fns.nns import NeuralNetwork
from src.models.dnns import DNN3
from src.models.linear_bayesian_models import (
    NeuralLinearModel,
    SpectralNormalizedNeuralGaussianProcess,
)
from src.utils.seeds import fix_random_seed
from src.utils.time_utils import timestamp


@dataclass
class Distribution:
    """No idea what this should be yet!"""


@dataclass
class MultivariateGaussian(Distribution):
    """Multivariate Gaussian object for Markovian settings, i.e. p(y) = \int p(x,y) dx = \int p(y | x) p(x) dx."""

    mean: torch.Tensor
    covariance: torch.Tensor
    cross_covariance: torch.Tensor
    previous_mean: torch.Tensor
    previous_covariance: torch.Tensor

    def marginalize(self, indices):
        """Marginalize out indices"""
        return MultivariateGaussian(
            self.mean[..., indices],
            self.covariance[..., indices, indices],
            None,
            None,
            None,
        )

    def full_joint(self, reverse=True):
        """Convert Markovian structure into the joint multivariate Gaussian and forget history."""
        if reverse:
            mean = torch.cat((self.previous_mean, self.mean), axis=-1)
            cross_covariance_T = einops.rearrange(
                self.cross_covariance, "... x1 x2 -> ... x2 x1"
            )
            covariance = torch.cat(
                (
                    torch.cat((self.previous_covariance, cross_covariance_T), axis=-1),
                    (torch.cat((self.cross_covariance, self.covariance), axis=-1)),
                ),
                axis=-2,
            )
        else:
            mean = torch.cat((self.mean, self.previous_mean), axis=-1)
            covariance = torch.cat(
                (
                    torch.cat((self.covariance, self.cross_covariance), axis=-1),
                    (
                        torch.cat(
                            (cross_covariance_T, self.previous_covariance), axis=-1
                        )
                    ),
                ),
                axis=-2,
            )
        return MultivariateGaussian(mean, covariance, None, None, None)

    def get_previous_loglikelihood(self):
        dim = self.previous_mean.shape[0]
        previous_precision = torch.linalg.inv(self.previous_covariance)

        def llh(x):
            diff = x - self.previous_mean
            return -0.5 * dim * (
                math.log(2.0 * math.pi) - torch.logdet(previous_precision)
            ) - 0.5 * torch.einsum("bi,ij,bj->b", diff, previous_precision, diff)

        return llh

    def reverse(self):
        return MultivariateGaussian(
            self.previous_mean,
            self.previous_covariance,
            self.cross_covariance.mT,  # transpose
            self.mean,
            self.covariance,
        )

    def sample(self, *args, **kwargs):
        """Works for batched mean and covariance"""
        return torch.distributions.MultivariateNormal(
            self.mean, self.covariance
        ).sample(*args, **kwargs)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.covariance = self.covariance.to(device)
        if self.cross_covariance is not None:
            self.cross_covariance = self.cross_covariance.to(device)
            self.previous_mean = self.previous_mean.to(device)
            self.previous_covariance = self.previous_covariance.to(device)
        return self

    def cpu(self):
        return self.to("cpu")


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
        return MultivariateGaussian(mean, covariance, None, None, None)


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
            # breakpoint()
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
        return MultivariateGaussian(mu_x, sigma_x, None, None, None)


def linear_gaussian_smoothing(
    current_prior: MultivariateGaussian,
    predicted_prior: MultivariateGaussian,
    future_posterior: MultivariateGaussian,
) -> MultivariateGaussian:
    """.
    See https://vismod.media.mit.edu/tech-reports/TR-531.pdf , Section 3.2
    """
    J = torch.linalg.solve(
        predicted_prior.covariance, predicted_prior.cross_covariance
    ).mT
    try:
        diff_mean = future_posterior.mean - predicted_prior.mean
    except:
        breakpoint()
    mean = current_prior.mean + einops.einsum(J, diff_mean, "... xu u, ... u -> ... xu")
    diff_cov = future_posterior.covariance - predicted_prior.covariance
    covariance = current_prior.covariance + J.matmul(diff_cov).matmul(J.mT)
    # if covariance.det() <= 0:  # TODO debug
    #     # covariance = nearest_spd(covariance)
    #     breakpoint()
    return MultivariateGaussian(
        mean,
        covariance,
        J.matmul(future_posterior.covariance),
        future_posterior.mean,
        future_posterior.covariance,
    )


class CudaAble(ABC):
    @abstractmethod
    def to(self, device):
        pass

    def cpu(self):
        self.to("cpu")


class Stateless:
    def to(self, device):
        pass

    def cpu(self):
        pass


@dataclass
class Model(CudaAble):
    approximate_inference: QuadratureInference
    """QuadratureInference.__call__:
    (x -> x_mu, x_cov, x_), MultivariateGaussian -> MultivariateGaussian
    => takes function that also returns (possibly) clamped input x_"""
    model: Callable
    """model.__call__: x, **_ -> ???"""

    def __call__(self, x, **kwargs):
        return self.model_call(x, **kwargs)

    def model_call(self, x, **kwargs):  # override to wrap call to model
        return self.model(x, **kwargs)

    def call_and_inputs(self, x, **kwargs):  # overload to use kwargs
        """calls model on input x and returns result and (unmodified) input"""
        res = self.model_call(x)
        x_ = x  # overload to modify inputs in model (e.g. action constraints)
        if type(res) in [tuple, list]:  # multiple return values
            return (*res, x_)
        else:
            return res, x_

    def propagate(self, dist: MultivariateGaussian, **kwargs) -> MultivariateGaussian:
        """propagate dist through model using approximate inference"""
        fun = partial(self.call_and_inputs, **kwargs)
        return self.approximate_inference(fun, dist)

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """predict with model (default: not stochastic)"""
        y, x_ = self.call_and_inputs(x, **kwargs)
        return y

    def to(self, device):
        self.model.to(device)
        self.approximate_inference.to(device)


@dataclass
class InputModifyingModel(Model):
    def call_and_inputs(self, x, **kwargs):  # overload to use kwargs
        *res, x_ = self.model_call(x)  # modifies x (e.g. action constraints)
        if type(res) in [tuple, list]:  # multiple return values
            return (*res, x_)
        else:
            return res, x_


@dataclass
class DeterministicModel(Model):
    pass  # identical to Model


@dataclass
class StochasticModel(Model):
    def predict(self, x: torch.Tensor, *, stoch=False, **kwargs) -> torch.Tensor:
        mu, cov, x_ = self.call_and_inputs(x, **kwargs)
        if stoch:
            return torch.distributions.MultivariateNormal(mu, cov).sample()
        else:
            return mu


@dataclass
class StochasticPolicy(StochasticModel):
    pass


@dataclass
class DeterministicPolicy(DeterministicModel):
    pass


@dataclass
class StochasticDynamics(StochasticModel, InputModifyingModel):
    pass


@dataclass
class DeterministicDynamics(DeterministicModel, InputModifyingModel):
    pass


@dataclass
class TimeVaryingStochasticPolicy(StochasticPolicy):
    def call_and_inputs(self, x, *, t: int, open_loop=False):
        mean, cov = self.model_call(x, t=t, open_loop=open_loop)
        x_ = x
        return mean, cov, x_


class TimeVaryingLinearGaussian(CudaAble):
    def __init__(
        self,
        horizon: int,
        dim_x: int,
        dim_u: int,
        action_covariance: torch.Tensor,
    ):
        assert action_covariance.shape == (dim_u, dim_u)
        self.horizon, self.dim_x, self.dim_u = horizon, dim_x, dim_u
        self.k_actual = torch.tile(
            1e-3 * torch.sin(torch.linspace(0, 20 * torch.pi, horizon))[:, None],
            (1, dim_u),
        )
        self.K_actual = torch.zeros((horizon, dim_u, dim_x))
        self.k_opt, self.K_opt = self.k_actual.clone(), self.K_actual.clone()
        self.sigma = torch.tile(action_covariance, (horizon, 1, 1))
        self.chol = torch.tile(
            torch.linalg.cholesky(action_covariance), (horizon, 1, 1)
        )

    def actual(self):
        from copy import deepcopy

        copy = deepcopy(self)  # crashes: only leaf tensors deepcopy-able
        # copy = TimeVaryingLinearGaussian(
        #     self.horizon,
        #     self.dim_x,
        #     self.dim_u,
        #     self.sigma[0, :, :].detach(),
        # )
        copy.k_opt = self.k_actual.detach()
        copy.K_opt = self.K_actual.detach()
        # TODO skip k/K_actual ?
        return copy

    def __call__(self, x: torch.Tensor, *, t: int, open_loop: bool = False):
        """Accepts (multi-)batch input."""
        # x: (b1, b2, ..., bm, x), K: (t, b1, b2, ..., bn, u, x)
        # k: (t, b1, ..., bn, u), sigma: (t, b1, ..., bn, u, u)
        if open_loop:
            K_t = self.K_opt[t, ...]
            k_t = self.k_opt[t, ...]
        else:
            K_t = self.K_actual[t, ...]
            k_t = self.k_actual[t, ...]
        # unsqueeze gains to match input batch size
        x_batch_dims = len(x.shape) - 1  # m: do not count x
        K_batch_dims = len(K_t.shape) - 2  # n: do not count u,x
        sigma = self.sigma[t, ...]
        for _ in range(x_batch_dims - K_batch_dims):  # for every extra batch dim in x
            K_t = K_t.unsqueeze(-2)  # add 1 dimension before u,x (so dims match)
            k_t = k_t.unsqueeze(-1)  # add 1 dimension before u (so dims match)
            sigma = sigma.unsqueeze(-2)  # add 1 dimension before u,u (so dims match)
        return (
            k_t + einops.einsum(K_t, x, "... u x, ... x -> ... u"),
            torch.tile(sigma, x.shape[K_batch_dims:-1] + (self.dim_u, self.dim_u)),
        )

    # def sample(self, t: int, x: torch.Tensor) -> torch.Tensor:
    #     breakpoint()  # broken!
    #     mu = self.k[t, :] + self.K[t, :, :] @ x
    #     eps = torch.randn(self.dim_u)
    #     return mu + self.chol[t, :, :] @ eps

    def update_from_distribution(self, distribution, *args, **kwargs):
        assert len(distribution) == self.horizon
        if isinstance(distribution[0], MultivariateGaussian):
            # recreate weights with correct batch dimensions
            device = distribution[0].mean.device
            batch_shape = [self.horizon] + list(distribution[0].mean.shape[:-1])
            bu = batch_shape + [self.dim_u]
            bux = batch_shape + [self.dim_u, self.dim_x]
            buu = batch_shape + [self.dim_u, self.dim_u]
            self.k_actual = torch.empty(bu, device=device)
            self.k_opt = torch.empty(bu, device=device)
            self.K_actual = torch.empty(bux, device=device)
            self.sigma = torch.empty(buu, device=device)
            self.chol = torch.empty(buu, device=device)
            # except K_opt: only unsqueeze (because is not updated = would be overwritten)
            K_opt = self.K_opt
            K_batch_dims = len(self.K_opt.shape) - 2  # do not cound u,x
            # for every extra batch dim
            for _ in range(len(batch_shape) - K_batch_dims):
                # add 1 dimension before u,x (so dims match)
                K_opt = K_opt.unsqueeze(-2)
            self.K_opt = K_opt
            for t, dist in enumerate(distribution):
                x = dist.marginalize(slice(0, self.dim_x))
                u = dist.marginalize(slice(self.dim_x, self.dim_x + self.dim_u))
                K = torch.linalg.solve(
                    x.covariance, dist.covariance[..., : self.dim_x, self.dim_x :]
                ).mT
                self.K_actual[t, ...] = K
                self.k_actual[t, ...] = u.mean - einops.einsum(
                    K, x.mean, "... u x, ... x -> ... u"
                )
                self.k_opt[t, ...] = u.mean - einops.einsum(
                    self.K_opt[t, ...], x.mean, "... u x, ... x -> ... u"
                )
                self.sigma[t, ...] = u.covariance
                self.chol[t, ...] = torch.linalg.cholesky(self.sigma[t, ...])
        else:
            raise ValueError("Cannot update from this distribution!")

    def to(self, device):
        self.k_actual = self.k_actual.to(device)
        self.K_actual = self.K_actual.to(device)
        self.k_opt = self.k_opt.to(device)
        self.K_opt = self.K_opt.to(device)
        self.sigma = self.sigma.to(device)
        self.chol = self.chol.to(device)
        return self


class Pendulum(Stateless):

    dim_x = 2
    dim_u = 1
    dim_xu = 3
    u_mx = 2.0

    def __call__(self, xu: torch.Tensor) -> torch.Tensor:
        dt = 0.05
        m = 1.0
        l = 1.0
        d = 1e-2  # damping
        g = 9.80665
        x, u = xu[..., :2], xu[..., 2]
        u = torch.clip(u, -self.u_mx, self.u_mx)
        th_dot_dot = (
            -3.0 * g / (2 * l) * torch.sin(x[..., 0] + torch.pi) - d * x[..., 1]
        )
        th_dot_dot += 3.0 / (m * l**2) * u
        # variant from http://underactuated.mit.edu/pend.html
        # th_dot_dot = -g / l * torch.sin(x[:, 0] + torch.pi) - d / (m*l**2) * x[:, 1]
        # th_dot_dot += 1.0 / (m * l**2) * u
        x_dot = x[..., 1] + th_dot_dot * dt  # theta_dot
        x_pos = x[..., 0] + x_dot * dt  # theta

        x2 = torch.stack((x_pos, x_dot), dim=-1)
        xu[..., 2] = u
        return x2, xu

    def run(self, initial_state, policy, horizon):
        batch_shape = [horizon] + list(initial_state.shape[:-1])
        xs = torch.zeros(batch_shape + [self.dim_x])
        us = torch.zeros(batch_shape + [self.dim_u])
        xxs = torch.zeros(batch_shape + [self.dim_x])
        state = initial_state
        for t in range(horizon):
            action = policy.predict(state, t=t)
            xu = torch.cat((state, action), dim=-1)
            xxs[t, ...], _ = self.__call__(xu)
            xs[t, ...] = state
            us[t, ...] = action
            state = xxs[t, ...]
        return xs, us, xxs

    def plot(self, xs, us):
        fig, axs = plt.subplots(self.dim_x + self.dim_u, figsize=(12, 9))
        xs = einops.rearrange(xs, "t ... x -> t (...) x")
        us = einops.rearrange(us, "t ... u -> t (...) u")
        n_batches = xs.shape[-2]
        for i in range(self.dim_x):
            for ib in range(n_batches):
                axs[i].plot(xs[:, ib, i])
            axs[i].set_ylabel(f"x{i}")
            axs[i].grid(True)
        for i in range(self.dim_u):
            j = self.dim_x + i
            for ib in range(n_batches):
                axs[j].plot(us[:, ib, i])
            axs[j].plot(self.u_mx * torch.ones_like(us[:, i]), "k--")
            axs[j].plot(-self.u_mx * torch.ones_like(us[:, i]), "k--")
            axs[j].set_ylabel(f"u{i}")
            axs[j].grid(True)
        axs[-1].set_xlabel("timestep")


def plot_gp(axis, mean, variance, color="b"):
    axis.plot(mean, color=color)
    sqrt = torch.sqrt(variance)
    upper, lower = mean + sqrt, mean - sqrt
    axis.fill_between(
        range(mean.shape[0]), upper, lower, where=upper >= lower, color=color, alpha=0.2
    )


def plot_trajectory_distribution(list_of_distributions, title=""):
    means = torch.stack(tuple(d.mean for d in list_of_distributions), dim=0)
    covariances = torch.stack(tuple(d.covariance for d in list_of_distributions), dim=0)
    # batch_shape = means.shape[1:-1]  # store batch dimensions
    # flatten batch dimensions (for plotting)
    means = einops.rearrange(means, "t ... x -> t (...) x")
    covariances = einops.rearrange(covariances, "t ... x1 x2 -> t (...) x1 x2")
    n_plots = means.shape[-1]
    n_batches = means.shape[1]
    colors = plt.cm.brg(np.linspace(0, 1, n_batches))
    fig, axs = plt.subplots(n_plots)
    axs[0].set_title(title)
    for i, ax in enumerate(axs):
        for b, c in zip(range(n_batches), colors):
            plot_gp(ax, means[:, b, i].cpu(), covariances[:, b, i, i].cpu(), color=c)
    return fig, axs


class PseudoPosteriorSolver(CudaAble):

    metrics: Dict

    def __init__(
        self,
        dim_x: int,
        dim_u: int,
        horizon: int,
        dynamics: InputModifyingModel,
        cost: Model,
        policy_template: TimeVaryingStochasticPolicy,
        smoother: Callable,
        update_temperature_strategy: Callable,
    ):
        self.dim_x, self.dim_u = dim_x, dim_u
        self.horizon = horizon
        self.dynamics = dynamics
        self.cost = cost
        self.policy_template = policy_template
        self.smoother = smoother
        self.update_temperature_strategy = update_temperature_strategy

    def forward_pass(
        self,
        policy: Model,
        initial_state: Distribution,
        alpha: float,
        open_loop: bool = True,
    ) -> (List[Distribution], Distribution):
        s = initial_state
        sa_cost_list = []
        next_s_list = []
        for t in range(self.horizon):
            a = policy.propagate(s, t=t, open_loop=open_loop)
            sa_policy = a.full_joint(reverse=True)
            sa_cost = self.cost.propagate(sa_policy, alpha=alpha)
            sa_cost_list += [sa_cost]
            # update state_action_cost with dynamics correlations
            next_s = self.dynamics.propagate(sa_cost)
            next_s_list += [next_s]
            s = next_s
        return sa_cost_list, next_s_list, s

    def backward_pass(
        self,
        forward_state_action_distribution,
        predicted_state_distribution,
        terminal_state_distribution,
    ):
        dist = []
        future_state_posterior = terminal_state_distribution
        for current_state_action_prior, pred_s_prior in zip(
            reversed(forward_state_action_distribution),
            reversed(predicted_state_distribution),
        ):
            state_action_posterior = self.smoother(
                current_state_action_prior,
                pred_s_prior,
                future_state_posterior,
            )
            # pass the state posterior to previous timestep
            future_state_posterior = state_action_posterior.marginalize(
                slice(0, self.dim_x)
            )
            dist += [state_action_posterior]
        return list(reversed(dist))

    def __call__(
        self,
        n_iteration: int,
        initial_state: Distribution,
        policy_prior: Model,
        plot_posterior: bool = True,
    ):
        # A1: tempstrat can be modified/tweaked
        # A2: forward to get filter is always on feedforward (also prior)
        self.init_metrics()
        # create as many alphas as batches (0:1 -> trailing 1 dimension)
        alpha = 0.0 * torch.ones_like(initial_state.mean[..., 0:1])
        policy_created = False
        if policy_prior:  # run once with prior to initialize policy
            policy = policy_prior
        else:  # create new (blank) policy
            policy = deepcopy(self.policy_template)
            policy_created = True

        for i in range(n_iteration):
            print(f"alpha {i}: {alpha.flatten().cpu()}")
            sa_filter, s_filter, s_T = self.forward_pass(
                policy,
                initial_state,
                alpha,
                open_loop=True,  # optimize only open-loop
            )
            sa_smoother = self.backward_pass(sa_filter, s_filter, s_T)
            # compute sa_policy trajectory (without cost: alpha=0)
            sa_policy_fb, _, _ = self.forward_pass(
                policy,
                initial_state,
                alpha=0.0,
                open_loop=False,  # to evaluate use feedback
            )
            self.compute_metrics(sa_smoother, sa_policy_fb, alpha)

            if not policy_created:  # switch from prior policy to local policy
                policy = deepcopy(self.policy_template)
                policy_created = True

            policy.model.update_from_distribution(sa_smoother)

            alpha = self.update_temperature_strategy(
                self.cost.predict,
                sa_smoother,  # unused for Polyak
                sa_policy_fb,
                current_alpha=alpha,  # unused for Polyak
            )
            if plot_posterior:
                # plot_trajectory_distribution(sa_filter, f"filter {i}")
                plot_trajectory_distribution(sa_smoother, f"posterior {i}")
                plt.show()
        return policy

    def compute_metrics(self, posterior_distribution, policy_distribution, alpha):
        posterior_cost = self.cost.predict(
            torch.stack(tuple(d.mean for d in posterior_distribution), dim=0)
        )[0].sum(dim=0)
        policy_cost = self.cost.predict(
            torch.stack(tuple(d.mean for d in policy_distribution), dim=0)
        )[0].sum(dim=0)
        self.metrics["posterior_cost"] += [posterior_cost.flatten().cpu()]
        self.metrics["policy_cost"] += [policy_cost.flatten().cpu()]
        self.metrics["alpha"] += [alpha.flatten().cpu()]

    def init_metrics(self):
        self.metrics = {
            "posterior_cost": [],
            "policy_cost": [],
            "alpha": [],
        }

    def plot_metrics(self):
        fix, axs = plt.subplots(len(self.metrics))
        for i, (k, v) in enumerate(self.metrics.items()):
            axs[i].plot(v.cpu())
            axs[i].set_ylabel(k)

    def to(self, device):
        self.dynamics.to(device)
        self.cost.to(device)
        self.policy_template.to(device)
        self.smoother.to(device)
        self.update_temperature_strategy.to(device)
        return self


class TemperatureStrategy(CudaAble):

    _alpha: torch.float32

    @abstractmethod
    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        pass


class Constant(TemperatureStrategy):
    """."""

    def __init__(self, alpha):
        assert alpha >= 0.0
        self._alpha = alpha

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        return self._alpha


class MaximumLikelihood(TemperatureStrategy):
    """For cost c = f(x, u), c \in [-\infty, 0], then the normalization term integrates to \exp(-\alpha c) / \alpha."""

    def __init__(self, inference: Callable):
        self.inference = inference

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        horizon = len(state_action_distribution)
        expected_cost = sum(
            [self.inference.mean(cost, dist) for dist in state_action_distribution]
        ).item()
        return horizon / expected_cost


class KullbackLeiblerDivergence(TemperatureStrategy):
    """."""

    def __init__(self, inference: Callable, epsilon: float):
        assert epsilon > 0.0
        self.inference = inference
        self.epsilon = epsilon

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        # fig, axs = plot_trajectory_distribution(state_action_distribution, "kl_constraint")
        # plt.show()
        horizon = len(state_action_distribution)
        # dim_xu = state_action_distribution[0].mean.shape[0]
        # state_action_next_state_distribution = [dist.full_joint(reverse=False) for dist in state_action_distribution]
        alpha_ = torch.ones((1,), requires_grad=True)
        # approximate integral \log\int p(x,u)\exp(-\alpha c(x,u)) dx du over trajectory distribution
        minimizer_args = dict(method="BFGS", options={"disp": True, "maxiter": 100})
        optimizer = MinimizeWrapper([alpha_], minimizer_args)
        # discretize trajectory into samples, weights and evaluations
        # traj_weights, traj_points = [], []
        # for dist in state_action_distribution:
        #     weights, points = inference.weights_mu, inference.get_x_pts(dist.mean, dist.covariance)
        #     traj_weights += [weights]
        #     traj_points += [points]
        # traj_costs = cost(torch.cat(traj_points, axis=0))
        # traj_weights = torch.cat(traj_weights, axis=0)
        list_of_weights, list_of_points = zip(
            *[
                (
                    self.inference.weights_mu,
                    self.inference.get_x_pts(dist.mean, dist.covariance),
                )
                for dist in state_action_distribution
            ]
        )
        # list_of_future_llh_fns = [dist.get_previous_loglikelihood() for dist in state_action_distribution]  # TODO naming sucks!!!!!
        # n_points = list_of_points[0].shape[0]
        # points are (horizon x n_points, dim_xu + dim_x)
        traj_weights, traj_points = map(torch.cat, [list_of_weights, list_of_points])
        traj_costs = cost(traj_points)
        non_zero_weight = traj_weights > 0.0
        log_traj_weights_nz = torch.log(traj_weights[non_zero_weight])
        # traj_future_state_llhs = torch.cat([llh(xunx[:, dim_xu:]) for (llh, xunx) in zip(list_of_future_llh_fns, list_of_points)])
        # import pdb; pdb.set_trace()
        # traj_current_state_llhs = torch.zeros_like(traj_future_state_llhs)
        # the first quadrature points are deterministic, so set weights to 1, otherwise shift future points
        # traj_current_state_llhs[n_points:] = traj_future_state_llhs[:-n_points]
        # import pdb; pdb.set_trace()
        alpha_.data = torch.Tensor(
            [1e-3]
        )  # horizon / torch.einsum("b,b->", traj_weights, traj_costs)
        # n = traj_weights.shape[0]
        # for i, ax in enumerate(axs):
        #     for t, pts in enumerate(list_of_points):
        #         ax.plot(t * torch.ones((pts.shape[0])), pts[: , i], 'g.')
        #         if i < 2:
        #             ax.plot(t * torch.ones((pts.shape[0])), pts[: , dim_xu + i], 'rx')
        advantage_nz = traj_costs[
            non_zero_weight
        ]  # + (traj_future_state_llhs[non_zero_weight] - traj_current_state_llhs[non_zero_weight]) / current_alpha
        alpha_max = 1000.0 / torch.abs(advantage_nz).min()
        print(alpha_max)
        # import pdb; pdb.set_trace()

        def dual(alpha: torch.float64):
            alpha = torch.clamp(torch.abs(alpha), max=alpha_max)
            # optimize advantage
            traj_alpha_cost_and_value = -alpha * advantage_nz
            # return self.epsilon / alpha + torch.log(sum([inference.mean(exp_alpha_cost, dist) for dist in state_action_distribution])) / alpha
            dual = (
                self.epsilon / alpha
                + torch.logsumexp(
                    log_traj_weights_nz + traj_alpha_cost_and_value, dim=0
                )
                / alpha
            )
            if torch.isinf(dual):
                import pdb

                pdb.set_trace()
            return dual

        def closure():
            optimizer.zero_grad()
            loss = dual(alpha_)
            loss.backward()
            return loss

        optimizer.step(closure)
        # import pdb; pdb.set_trace()
        return abs(alpha_.item())


class PolyakStepSize(TemperatureStrategy):
    """."""

    def __init__(self, inference: Callable):
        assert isinstance(
            inference, QuadratureInference
        ), "Computation requires sample discretization"
        self.inference = inference

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,  # unused
        state_action_policy,
        current_alpha,  # unused
    ):
        """ " batched version (batch dims first)"""
        # list_of_weights, list_of_costs = zip(
        #     *[
        #         (
        #             self.inference.weights_mu,
        #             cost(self.inference.get_x_pts(dist.mean, dist.covariance))[0],
        #         )
        #         for dist in state_action_policy
        #     ]
        # )
        # breakpoint()
        # traj_weights, traj_costs = map(torch.cat, [list_of_weights, list_of_costs])
        # sup_cost = sum([torch.max(costs_t) for costs_t in list_of_costs])
        # expected_cost = torch.einsum("b,b->", traj_weights, traj_costs)
        list_of_costs = []
        list_of_sup_costs = []
        for dist in state_action_policy:
            x_t_pts = self.inference.get_x_pts(dist.mean, dist.covariance)
            cost_t_pts = cost(x_t_pts)
            list_of_costs.append(cost_t_pts)
            max_cost_t, _ = cost_t_pts.max(dim=-1)
            list_of_sup_costs.append(max_cost_t)
        traj_costs = torch.stack(list_of_costs, dim=-1)
        sup_cost = torch.stack(list_of_sup_costs, dim=-1).sum(dim=-1)
        expected_cost = einops.einsum(
            self.inference.weights_mu, traj_costs, "p, ... p time -> ..."
        )
        return einops.rearrange(expected_cost / sup_cost, "... -> ... 1")

    def to(self, device):
        self.inference.to(device)
        return self


class QuadraticModel(TemperatureStrategy):
    def __init__(self, inference):
        self.inference = inference
        self.previous_expected_cost = None

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        expected_cost_posterior = sum(
            [self.inference.mean(cost, dist) for dist in state_action_distribution]
        )
        expected_cost_policy = sum(
            [self.inference.mean(cost, dist) for dist in state_action_policy]
        )
        if self.previous_expected_cost is None:
            self.previous_expected_cost = expected_cost_posterior
        new_alpha = (
            current_alpha
            * (self.previous_expected_cost - expected_cost_posterior)
            / (2 * (self.previous_expected_cost - expected_cost_policy) + 1e-6)
        )
        self.previous_expected_cost = expected_cost_policy
        return new_alpha


class Annealing(TemperatureStrategy):
    def __init__(self, start, end, iterations):
        assert start > 0.0
        assert end > start
        # self.schedule = iter(torch.exp(torch.linspace(math.log(start), math.log(end), iterations)).tolist())
        self.schedule = iter(
            (
                start
                + (end - start)
                * torch.sqrt(torch.linspace(0, iterations, iterations))
                / math.sqrt(iterations)
            ).tolist()
        )

    def __call__(
        self,
        cost: Callable,
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        return next(self.schedule)


class HitCounter:
    spd_counter = 0  # static

    def spd_fix():
        HitCounter.spd_counter += 1


def nearest_spd(covariance):
    return covariance  # do nothing
    HitCounter.spd_fix()
    L, V = torch.linalg.eig(covariance)
    L[L.real <= 1e-10] = 1e-6  # is it safe to do real?
    return V @ torch.diag(L) @ V.T


def compose(wrapper, func):
    @functools.wraps(func)
    def composed(*args, **kwargs):
        return wrapper(func(*args, **kwargs))

    return composed


def experiment(
    env_type: str = "localPendulum",  # Pendulum
    horizon: int = 200,
    n_rollout_episodes: int = 10,
    batch_size: int = 200 * 10,  # lower if gpu out of memory
    # plot_data: bool = False,
    n_iter: int = 10,  # outer loop
    use_cuda: bool = True,  # for policy/dynamics training (i2c on cpu)
    log_frequency: float = 0.1,  # every p% epochs of n_epochs
    model_save_frequency: float = 0.5,  # every n-th of n_epochs
    ## dynamics ##
    plot_dyn: bool = True,  # plot pointwise and rollout prediction
    # #  D1) true dynamics
    # dyn_model_type: str = "env",
    # #  D2) dnn model
    # dyn_model_type: str = "mlp",
    # n_hidden_dyn: int = 256,
    # n_hidden_layers_dyn: int = 2,  # 2 ~ [in, h, h, out]
    # lr_dyn: float = 3e-4,
    # n_epochs_dyn: int = 1000,
    # # D3) linear regression w/ dnn features
    # dyn_model_type: str = "nlm",
    # n_features_dyn: int = 128,
    # n_hidden_dyn: int = 128,
    # n_hidden_layers_dyn: int = 2,  # 2 ~ [in, h, h, out]
    # lr_dyn: float = 1e-4,
    # n_epochs_dyn: int = 1000,
    # D4) linear regression w/ sn-dnn & rf features
    dyn_model_type: str = "snngp",  # TODO why so slow? (x15) paper says x1.2
    n_features_dyn: int = 128,  # RFFs require ~512-1024 for accuracy
    n_hidden_dyn: int = 128,
    n_hidden_layers_dyn: int = 2,  # 2 ~ [in, h, h, out]
    lr_dyn: float = 1e-4,
    n_epochs_dyn: int = 1000,
    ##############
    ## policy ##
    plot_policy: bool = False,  # plot pointwise and rollout prediction
    #  D1) local time-varying linear gaussian controllers (i2c)
    policy_type: str = "tvlg",
    # #  D2) dnn model
    # policy_type: str = "mlp",
    # n_hidden_pol: int = 256,
    # n_hidden_layers_pol: int = 2,  # 2 ~ [in, h, h, out]
    # lr_pol: float = 3e-4,
    # n_epochs_pol: int = 100,
    ############
    ## i2c solver ##
    n_iter_solver: int = 5,  # how many i2c solver iterations to do
    n_i2c_local_policies: int = 10,  # how many local policies in the mixture i2c
    plot_posterior: bool = False,  # plot state-action-posterior over time
    # plot_posterior: bool = True,  # plot state-action-posterior over time
    plot_local_policy_metrics: bool = False,  # plot time-cum. sa-posterior cost, local policy cost, and alpha per iter
    # plot_local_policy_metrics: bool = True,  # plot time-cum. sa-posterior cost, local policy cost, and alpha per iter
    ############
    ## general ##
    plotting: bool = True,  # if False overrides all other flags
    plot_data: bool = True,  # visualize data trajectories (sns => very slow!)
    log_console: bool = True,  # also log to console (not just log file); FORCE on if debug
    log_wandb: bool = True,  # off if debug
    wandb_project: str = "smbrl_i2c",
    wandb_entity: str = "showmezeplozz",
    wandb_job_type: str = "train",
    seed: int = 0,
    results_dir: str = "logs/tmp/",
    debug: bool = True,  # prepends logdir with 'debug/', disables wandb, enables console logging
    # debug: bool = False,  # prepends logdir with 'debug/', disables wandb, enables console logging
):
    ####################################################################################################################
    #### SETUP (saved to yaml)

    if debug:
        # disable wandb logging and redirect normal logging to ./debug directory
        print("@@@@@@@@@@@@@@@@@ DEBUG: LOGGING DISABLED @@@@@@@@@@@@@@@@@")
        os.environ["WANDB_MODE"] = "disabled"  # close terminal to reset
        log_wandb = False
        log_console = True
        results_dir = "debug" / Path(results_dir)

    # Fix seed
    fix_random_seed(seed)

    # Results directory
    wandb_group: str = f"i2c_{env_type}_{dyn_model_type}_{policy_type}"
    repo_dir = Path.cwd().parent
    results_dir = repo_dir / results_dir / wandb_group / str(seed) / timestamp()
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # Save arguments
    save_args(results_dir, locals(), git_repo_path="./")

    # logger
    logger = Logger(
        config=locals(),
        log_name=seed,
        results_dir=results_dir,
        project=wandb_project,
        entity=wandb_entity,
        wandb_kwargs={"group": wandb_group, "job_type": wandb_job_type},
        tags=["dyn_model_type", "policy_type", "env_type"],
        log_console=log_console,
        log_wandb=log_wandb,
    )

    ####################################################################################################################
    #### EXPERIMENT SETUP

    logger.info(f"Env: {env_type}, Dyn: {dyn_model_type}, Pol: {policy_type}")
    logger.info(f"Seed: {seed}")

    time_begin = time.time()

    # visialization options
    torch.set_printoptions(precision=7)
    torch.set_default_dtype(torch.float64)

    ### mdp, initial state, cost ###
    if env_type == "localPendulum":
        environment = Pendulum()  # local seed
    dim_xu = environment.dim_xu
    dim_x = environment.dim_x
    dim_u = environment.dim_u
    initial_state = torch.Tensor([torch.pi, 0.0])
    initial_state_distribution = MultivariateGaussian(
        initial_state,
        # 1e-6 * torch.eye(dim_x),
        1e-2 * torch.eye(dim_x),  # more exploration
        None,
        None,
        None,
    )

    dyn_train_buffer = ReplayBuffer(
        dim_xu, dim_x, batchsize=batch_size, device=device, max_size=1e4
    )
    dyn_test_buffer = ReplayBuffer(
        dim_xu, dim_x, batchsize=batch_size, device=device, max_size=1e4
    )

    ### approximate inference params ###
    quad_params = CubatureQuadrature(1, 0, 0)
    gh_params = GaussHermiteQuadrature(degree=3)
    # if deterministic -> QuadratureInference (unscented gaussian approx)
    # if gaussian -> QuadratureGaussianInference (unscented gaussian approx)

    def sincos_angle(xu):
        # input angles get split sin/cos
        xu_sincos_shape = list(xu.shape)
        xu_sincos_shape[-1] += 1  # add another x dimension
        xu_sincos = torch.empty(xu_sincos_shape, device=xu.device)
        xu_sincos[..., 0] = xu[..., 0].sin()
        xu_sincos[..., 1] = xu[..., 0].cos()
        xu_sincos[..., 2] = xu[..., 1]
        xu_sincos[..., 3] = xu[..., 2]
        return xu_sincos

    ### cost (model) ###
    @dataclass
    class CostModel(DeterministicModel):
        model: Callable = None  # unused in model_call
        approximate_inference: QuadratureImportanceSamplingInnovation

        def model_call(self, xu, **kwargs):
            """batched version (batch dims first)"""
            # swing-up: \theta = \pi -> 0
            theta, theta_dot, u = xu[..., 0], xu[..., 1], xu[..., 2]
            theta_cost = (torch.cos(theta) - 1.0) ** 2
            theta_dot_cost = 1e-2 * theta_dot**2
            u_cost = 1e-2 * u**2
            total_cost = theta_cost + theta_dot_cost + u_cost
            return total_cost

        def propagate(
            self, dist: MultivariateGaussian, *, alpha, **kwargs
        ) -> MultivariateGaussian:
            fun = partial(self.call_and_inputs, **kwargs)
            return self.approximate_inference(fun, alpha, dist)

    cost = CostModel(
        approximate_inference=QuadratureImportanceSamplingInnovation(
            dim_xu,
            gh_params,
        ),
    )

    ### global dynamics model ###
    if dyn_model_type == "env":
        global_dynamics = DeterministicDynamics(
            model=environment,
            approximate_inference=QuadratureInference(dim_xu, quad_params),
        )
    elif dyn_model_type == "mlp":

        @dataclass
        class MLPDynamics(DeterministicDynamics):
            def model_call(self, xu, **kwargs):
                x, u = xu[..., :dim_x], xu[..., dim_x]
                u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
                xu_sincos = sincos_angle(xu)
                delta_x = self.model(xu_sincos)  # output states are deltas
                return x.detach() + delta_x, xu

        global_dynamics = MLPDynamics(
            model=NeuralNetwork(
                dim_xu + 1,  # sin,cos of theta
                n_hidden_layers_dyn,
                n_hidden_dyn,
                dim_x,
            ),
            approximate_inference=QuadratureInference(dim_xu, quad_params),
        )

        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)
        loss_fn_dyn = lambda x, y: torch.nn.MSELoss()(global_dynamics.predict(x), y)
        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)
    elif dyn_model_type == "nlm":

        @dataclass
        class NLMDynamics(StochasticDynamics):
            def model_call(self, xu, **kwargs):
                x, u = xu[..., :dim_x], xu[..., dim_x]
                u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
                xu_sincos = sincos_angle(xu)
                # output states are deltas
                delta_mu, cov, cov_in, cov_out = self.model(xu_sincos)
                # cov_bij = einops.einsum(cov_out, cov_in, "o o, ... i1 i2 -> ... o i1 i2")
                cov_bij = einops.einsum(
                    cov_out, cov_in, "o1 o2, ... i i -> ... i o1 o2"
                )
                mu = x.detach() + delta_mu
                return mu, cov_bij, xu

        global_dynamics = NLMDynamics(
            model=NeuralLinearModel(
                dim_xu + 1,  # sin,cos of theta
                dim_x,
                n_hidden_dyn,
                n_features_dyn,
                n_hidden_layers_dyn,
            ),
            approximate_inference=QuadratureGaussianInference(dim_xu, quad_params),
        )

        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)

        def loss_fn_dyn(xu, x_next):
            x, u = xu[..., :dim_x], xu[..., dim_x]
            u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
            xu_sincos = sincos_angle(xu)
            delta_x = x_next - x
            return -global_dynamics.model.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)
    elif dyn_model_type == "snngp":

        @dataclass
        class SNNGPDynamics(StochasticDynamics):
            def model_call(self, xu, **kwargs):
                x, u = xu[..., :dim_x], xu[..., dim_x]
                u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
                xu_sincos = sincos_angle(xu)
                # output states are deltas
                delta_mu, cov, cov_in, cov_out = self.model(xu_sincos)
                # cov_bij = einops.einsum(cov_out, cov_in, "o o, ... i1 i2 -> ... o i1 i2")
                cov_bij = einops.einsum(
                    cov_out, cov_in, "o1 o2, ... i i -> ... i o1 o2"
                )
                mu = x.detach() + delta_mu
                return mu, cov_bij, xu

        global_dynamics = SNNGPDynamics(
            model=SpectralNormalizedNeuralGaussianProcess(
                dim_xu + 1,  # sin,cos of theta
                dim_x,
                n_hidden_dyn,
                n_features_dyn,
                n_hidden_layers_dyn,
            ),
            approximate_inference=QuadratureGaussianInference(dim_xu, quad_params),
        )

        # global_dynamics.model.init_whitening(dyn_train_buffer.xs, dyn_train_buffer.ys)

        def loss_fn_dyn(xu, x_next):
            x, u = xu[..., :dim_x], xu[..., dim_x]
            u = torch.clamp(u, -environment.u_mx, +environment.u_mx)
            xu_sincos = sincos_angle(xu)
            delta_x = x_next - x
            return -global_dynamics.model.elbo(xu_sincos, delta_x)

        opt_dyn = torch.optim.Adam(global_dynamics.model.parameters(), lr=lr_dyn)

    ### local (i2c) policy ###
    local_policy = TimeVaryingStochasticPolicy(
        model=TimeVaryingLinearGaussian(
            horizon,
            dim_x,
            dim_u,
            action_covariance=0.2 * torch.eye(dim_u),
        ),
        approximate_inference=QuadratureGaussianInference(dim_x, quad_params),
    )

    ### global policy model ###
    if policy_type == "tvlg":
        global_policy = deepcopy(local_policy)
    elif policy_type == "mlp":
        global_policy = DeterministicPolicy(
            model=NeuralNetwork(
                dim_x,
                n_hidden_layers_pol,
                n_hidden_pol,
                dim_u,
            ),
            approximate_inference=QuadratureInference(dim_x, quad_params),
        )
        # global_policy.model.init_whitening(train_buffer.xs, train_buffer.ys)
        loss_fn_pol = lambda x, y: torch.nn.MSELoss()(global_policy.predict(x), y)
        opt_pol = torch.optim.Adam(global_policy.model.parameters(), lr=lr_pol)
    elif policy_type == "nlm":
        pass  # TODO
        approx_inf = QuadratureGaussianInference(dim_x, quad_params)
    elif policy_type == "snngp":
        pass  # TODO
        approx_inf = QuadratureGaussianInference(dim_x, quad_params)

    ### i2c solver ###
    i2c_solver = PseudoPosteriorSolver(
        dim_x=dim_x,
        dim_u=dim_u,
        horizon=horizon,
        dynamics=global_dynamics,
        cost=cost,
        policy_template=local_policy,
        smoother=linear_gaussian_smoothing,
        # update_temperature_strategy=Constant(alpha),
        # update_temperature_strategy=MaximumLikelihood(QuadratureInference(dim_xu, quad_params)),
        # update_temperature_strategy=QuadraticModel(QuadratureInference(dim_xu, quad_params)),
        # update_temperature_strategy=KullbackLeiblerDivergence(QuadratureInference(dim_xu, gh_params), epsilon=kl_bound),
        # update_temperature_strategy=Annealing(1e-2, 5e1, 15+1),
        update_temperature_strategy=PolyakStepSize(
            QuadratureInference(dim_xu, quad_params)
        ),
    )

    ####################################################################################################################
    #### TRAINING

    # for alpha in [1e-1, 1, 10, 100, 1000]:
    # for kl_bound in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    # for _ in [None]:
    dyn_loss_trace = []
    dyn_test_loss_trace = []
    pol_loss_trace = []
    pol_test_loss_trace = []
    for i_iter in range(n_iter):
        logger.strong_line()
        logger.info(f"ITERATION {i_iter + 1}/{n_iter}")
        if dyn_model_type != "env":
            global_dynamics.cpu()  # in-place
            torch.set_grad_enabled(False)

        #### T: collect dynamics rollouts
        # # global policy (tvlg with feedback)
        # exploration_policy = compose(lambda _: _[0], global_policy)

        # # tvlg (only!!!) without feedback
        # exploration_policy = compose(
        #     lambda _: _[0],
        #     partial(global_policy.__call__, with_feedback=False)
        # )

        # 50-50 %: tvgl-fb or gaussian noise

        def add_dithering(y):
            return y + torch.randn_like(y)

            # # if torch.randn(1) > 0.5:
            # #     # return global_policy.predict(x, t)  # TODO actual makes no difference?
            # #     return global_policy.actual().predict(x, t) + dithering
            # # else:
            # #     return torch.normal(torch.zeros(dim_u), 3 * torch.ones(dim_u))
            # return global_policy.predict(x, **kwargs) + dithering
            # # return torch.normal(0.0 * torch.ones(dim_u), 1e-2 * torch.ones(dim_u))

        # # random gaussian
        # exploration_policy.predict = lambda *_: torch.randn(dim_u)  # N(0,1)
        # exploration_policy.predict = lambda *_: torch.randn(dim_u) * 3

        exploration_policy = deepcopy(global_policy)
        exploration_policy.predict = compose(add_dithering, exploration_policy.predict)

        # train and test rollouts (env & exploration policy)
        logger.weak_line()
        logger.info("START Collecting Rollouts")
        for i in trange(n_rollout_episodes):  # 80 % train
            state = initial_state_distribution.sample()
            s, a, ss = environment.run(state, exploration_policy, horizon)
            dyn_train_buffer.add(torch.hstack([s, a]), ss)
        for i in trange(int(n_rollout_episodes / 4)):  # 20 % test
            state = initial_state_distribution.sample()
            s, a, ss = environment.run(state, exploration_policy, horizon)
            dyn_test_buffer.add(torch.hstack([s, a]), ss)
        logger.info("END Collecting Rollouts")

        #### T: train dynamics model
        if dyn_model_type != "env":
            # Learn Dynamics
            logger.weak_line()
            logger.info("START Training Dynamics")
            global_dynamics.to(device)  # in-place
            torch.set_grad_enabled(True)

            ## initial loss
            # for minibatch in dyn_train_buffer:  # TODO not whole buffer!
            #     _x, _y = minibatch
            #     _loss = loss_fn_dyn(_x, _y)
            #     dyn_loss_trace.append(_loss.detach().item())
            #     logger.log_data(
            #         **{
            #             "dynamics/train/loss": dyn_loss_trace[-1],
            #         },
            #     )
            # test_losses = []
            # for minibatch in dyn_test_buffer:  # TODO not whole buffer!
            #     _x_test, _y_test = minibatch
            #     _test_loss = loss_fn_dyn(_x_test, _y_test)
            #     test_losses.append(_test_loss.detach().item())
            # dyn_test_loss_trace.append(np.mean(test_losses))
            # logger.log_data(
            #     step=logger._step,  # in sync with training loss
            #     **{
            #         "dynamics/eval/loss": dyn_test_loss_trace[-1],
            #     },
            # )

            # torch.autograd.set_detect_anomaly(True)
            for i_epoch_dyn in trange(n_epochs_dyn + 1):
                for i_minibatch, minibatch in enumerate(dyn_train_buffer):
                    x, y = minibatch
                    opt_dyn.zero_grad()
                    loss = loss_fn_dyn(x, y)
                    loss.backward()
                    opt_dyn.step()
                    dyn_loss_trace.append(loss.detach().item())
                    logger.log_data(
                        **{
                            # "dynamics/train/epoch": i_epoch_dyn,
                            "dynamics/train/loss": dyn_loss_trace[-1],
                        },
                    )

                # save logs & test
                logger.save_logs()
                if i_epoch_dyn % (n_epochs_dyn * log_frequency) == 0:
                    with torch.no_grad():
                        # test loss
                        dyn_test_buffer.shuffling = False  # TODO only for plotting?
                        test_losses = []
                        for minibatch in dyn_test_buffer:  # TODO not whole buffer!
                            _x_test, _y_test = minibatch
                            _test_loss = loss_fn_dyn(_x_test, _y_test)
                            test_losses.append(_test_loss.item())
                        dyn_test_loss_trace.append(np.mean(test_losses))
                        logger.log_data(
                            step=logger._step,  # in sync with training loss
                            **{
                                "dynamics/eval/loss": dyn_test_loss_trace[-1],
                            },
                        )

                        logstring = f"DYN: Epoch {i_epoch_dyn}, Train Loss={dyn_loss_trace[-1]:.2}"
                        logstring += f", Test Loss={dyn_test_loss_trace[-1]:.2}"
                        logger.info(logstring)

                # stop condition
                if i_epoch_dyn > 0.01 * n_epochs_dyn:  # do at least 1% of epochs
                    if dyn_test_loss_trace[-1] < -3e3:  # stop if test loss good
                        break

                # TODO save model more often than in each global iter?
                # if n % (n_epochs * model_save_frequency) == 0:
                #     # Save the agent
                #     torch.save(model.state_dict(), results_dir / f"agent_{n}_{i_iter}.pth")

            # Save the model after training
            global_dynamics.cpu()  # in-place
            torch.set_grad_enabled(False)
            torch.save(
                global_dynamics.model.state_dict(),
                results_dir / "dyn_model_{i_iter}.pth",
            )
            logger.info("END Training Dynamics")

            #### T: plot dynamics model
            if plot_dyn:
                ## test dynamics model in rollouts
                # TODO extract?
                ## data traj (from buffer)
                # sa_env = dyn_test_buffer.xs[:horizon, :].cpu()  # first
                sa_env = dyn_test_buffer.xs[-horizon:, :].cpu()  # last
                s_env, a_env = sa_env[:, :dim_x], sa_env[:, dim_x:]
                # ss_env = dyn_test_buffer.ys[:horizon, :].cpu()  # first
                ss_env = dyn_test_buffer.ys[-horizon:, :].cpu()  # last
                ss_pred_pw = torch.zeros((horizon, dim_x))
                ss_pred_roll = torch.zeros((horizon, dim_x))
                state = s_env[0, :]  # for rollouts: data init state
                for t in range(horizon):
                    # pointwise
                    xu = torch.cat((s_env[t, :], a_env[t, :]))[None, :]  # pred traj
                    if dyn_model_type in ["nlm", "snngp"]:
                        ss_pred_pw[t, :], var, xu_ = global_dynamics(xu)
                    else:
                        ss_pred_pw[t, :], xu_ = global_dynamics(xu)
                    # rollout (replay action)
                    xu = torch.cat((state, a_env[t, :]))[None, :]
                    if dyn_model_type in ["nlm", "snngp"]:
                        ss_pred_roll[t, :], var, xu = global_dynamics(xu)
                    else:
                        ss_pred_roll[t, :], xu = global_dynamics(xu)
                    state = ss_pred_roll[t, :]
                # compute costs (except init state use pred next state)
                c_env = cost.predict(sa_env)
                s_pred_pw = torch.cat([s_env[:1, :], ss_pred_pw[:-1, :]])
                sa_pred_pw = torch.cat([s_pred_pw, a_env], dim=1)
                c_pw = cost.predict(sa_pred_pw)
                s_pred_roll = torch.cat([s_env[:1, :], ss_pred_roll[:-1, :]])
                sa_pred_roll = torch.cat([s_pred_roll, a_env], dim=1)
                c_roll = cost.predict(sa_pred_roll)

                ### plot pointwise and rollout predictions (1 episode) ###
                fig, axs = plt.subplots(dim_u + 1 + dim_x, 2, figsize=(10, 7))
                steps = torch.tensor(range(0, horizon))
                axs[0, 0].set_title("pointwise predictions")
                axs[0, 1].set_title("rollout predictions")
                # plot actions (twice: left & right)
                for ui in range(dim_u):
                    axs[ui, 0].plot(steps, a_env[:, ui], color="b")
                    axs[ui, 0].set_ylabel("action")
                    axs[ui, 1].plot(steps, a_env[:, ui], color="b")
                    axs[ui, 1].set_ylabel("action")
                # plot cost
                ri = dim_u
                axs[ri, 0].plot(steps, c_env, color="b", label="data")
                axs[ri, 0].plot(steps, c_pw, color="r", label=dyn_model_type)
                axs[ri, 0].set_ylabel("cost")
                axs[ri, 1].plot(steps, c_env, color="b", label="data")
                axs[ri, 1].plot(steps, c_roll, color="r", label=dyn_model_type)
                axs[ri, 1].set_ylabel("cost")
                for xi in range(dim_x):
                    xi_ = xi + dim_u + 1  # plotting offset
                    # plot pointwise state predictions
                    axs[xi_, 0].plot(steps, ss_env[:, xi], color="b")
                    axs[xi_, 0].plot(steps, ss_pred_pw[:, xi], color="r")
                    axs[xi_, 0].set_ylabel(f"ss[{xi}]")
                    # plot rollout state predictions
                    axs[xi_, 1].plot(steps, ss_env[:, xi], color="b")
                    axs[xi_, 1].plot(steps, ss_pred_roll[:, xi], color="r")
                    axs[xi_, 1].set_ylabel(f"ss[{xi}]")
                axs[-1, 0].set_xlabel("steps")
                axs[-1, 1].set_xlabel("steps")
                handles, labels = axs[ri, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=2)
                fig.suptitle(
                    f"{dyn_model_type} pointwise and rollout dynamics on 1 episode "
                    f"({int(dyn_test_buffer.size/horizon)} "
                    f"episodes, {i_iter * n_epochs_dyn} epochs, lr={lr_dyn})"
                )
                plt.savefig(results_dir / f"dyn_eval_{i_iter}.png", dpi=150)
                if plotting:
                    plt.show()

        #### T: i2c
        # i2c: find local (optimal) tvlg policy
        logger.weak_line()
        logger.info(f"START i2c [{n_iter_solver} iters]")
        # sample initial states for i2c mixture (batched init_state_dist)
        init_mean = initial_state_distribution.sample([n_i2c_local_policies])
        init_covar = 1e-6 * torch.eye(dim_x).repeat(n_i2c_local_policies, 1, 1)
        s0_dist = MultivariateGaussian(init_mean, init_covar, None, None, None)

        local_mixture_policy = i2c_solver(
            n_iteration=n_iter_solver,
            initial_state=s0_dist,
            policy_prior=global_policy,
            plot_posterior=plot_posterior and plotting,
        )
        logger.info("END i2c")
        # log i2c metrics
        for i_ in range(n_iter_solver):
            log_dict = {"iter": i_iter, "i2c_iter": i_}
            for (k, v) in i2c_solver.metrics.items():
                log_dict[k] = v[i_]  # one key, n_iter_solver values
            logger.log_data(log_dict)

        #### T: plot i2c opt. controller
        ## plot current i2c optimal controller
        if plot_local_policy_metrics and plotting:
            fix, axs = plt.subplots(3)
            for i, (k, v) in enumerate(i2c_solver.metrics.items()):
                v = torch.tensor(v)
                # axs[i].plot(v, label=f"$\epsilon={kl_bound}$")
                # axs[i].plot(v, label=f"$\\alpha={alpha}$")
                axs[i].plot(v, label=f"$Polyak$")
                # axs[i].plot(v, label=f"Maximum Likelihood")
                # axs[i].plot(v, label="Quadratic Model")
                # axs[i].plot(v, label="Annealing")
                axs[i].set_ylabel(k)
            for ax in axs:
                ax.legend()
            # TODO clean up plot
            plt.savefig(results_dir / "i2c_metrics_{i_iter}.png", dpi=150)
            plt.show()

        # Fit global policy to local policy
        if policy_type == "tvlg":
            global_policy = deepcopy(local_policy)
            # what makes more sense?
            # # a) average over gains
            # global_policy.model.k_actual = local_mixture_policy.model.k_actual.mean(1)
            # global_policy.model.K_actual = local_mixture_policy.model.K_actual.mean(1)
            # global_policy.model.k_opt = local_mixture_policy.model.k_opt.mean(1)
            # global_policy.model.K_opt = local_mixture_policy.model.K_opt.mean(1)
            # b) take first gains
            global_policy.model.k_actual = local_mixture_policy.model.k_actual[:, 0, :]
            global_policy.model.K_actual = local_mixture_policy.model.K_actual[:, 0, :]
            global_policy.model.k_opt = local_mixture_policy.model.k_opt[:, 0, :]
            global_policy.model.K_opt = local_mixture_policy.model.K_opt[:, 0, :]
        else:
            #### T: collect policy rollouts
            # Roll out local policy mixture to get samples to train global policy
            breakpoint()
            # TODO

            # min KL[mean local_policy || global_policy]
            # should do: global_policy.predict(x, t)
            logger.weak_line()
            logger.info("START Training Policy")
            global_policy.to(device)  # in-place
            torch.set_grad_enabled(True)

            #### T: train policy
            for i_epoch_pol in trange(n_epochs_pol + 1):
                for i_minibatch, minibatch in enumerate(train_buffer):
                    x, y = minibatch
                    opt_pol.zero_grad()
                    loss = loss_fn_pol(x, y)
                    loss.backward()
                    opt_dyn.step()
                    pol_loss_trace.append(loss.detach().item())
                    logger.log_data(
                        **{
                            # "policy/train/epoch": i_epoch_dyn,
                            "policy/train/loss": pol_loss_trace[-1],
                        },
                    )

                # save logs & test
                logger.save_logs()
                if i_epoch_pol % (n_epochs_pol * log_frequency) == 0:
                    with torch.no_grad():
                        # test loss
                        test_buffer.shuffling = False  # TODO only for plotting?
                        test_losses = []
                        for minibatch in test_buffer:  # TODO not whole buffer!
                            _x_test, _y_test = minibatch
                            _test_loss = loss_fn_pol(_x_test, _y_test)
                            test_losses.append(_test_loss.item())
                        pol_test_loss_trace.append(np.mean(test_losses))
                        logger.log_data(
                            step=logger._step,  # in sync with training loss
                            **{
                                "policy/eval/loss": pol_test_loss_trace[-1],
                            },
                        )

                        logstring = f"pol: Epoch {i_epoch_pol}, Train Loss={pol_loss_trace[-1]:.2}"
                        logstring += f", Test Loss={pol_test_loss_trace[-1]:.2}"
                        logger.info(logstring)

                # stop condition
                if i_epoch_pol > 0.01 * n_epochs_pol:  # do at least 1% of epochs
                    if pol_test_loss_trace[-1] < -3e3:  # stop if test loss good
                        break

                # TODO save model more often than in each global iter?
                # if n % (n_epochs * model_save_frequency) == 0:
                #     # Save the agent
                #     torch.save(model.state_dict(), results_dir / f"agent_{n}_{i_iter}.pth")

            # Save the model after training
            global_policy.cpu()  # in-place
            torch.set_grad_enabled(False)
            torch.save(
                global_policy.model.state_dict(),
                results_dir / "pol_model_{i_iter}.pth",
            )
            logger.info("END Training policy")

            #### T: plot policy
            if plot_policy:
                pass  # TODO

    ####################################################################################################################
    #### EVALUATION

    # local_policy.plot_metrics()
    initial_state = torch.Tensor([torch.pi, 0.0])
    # initial_state = torch.Tensor([torch.pi + 0.4, 0.0])  # breaks local_policy!!

    ### policy vs env
    # TODO label plot
    # TODO sample action?
    xs, us, xxs = environment.run(initial_state, global_policy, horizon)
    environment.plot(xs, us)
    plt.suptitle(f"{policy_type} policy vs env")
    plt.savefig(results_dir / f"{policy_type}_vs_env_{i_iter}.png", dpi=150)

    ### policy vs dyn model
    xs = torch.zeros((horizon, dim_x))
    us = torch.zeros((horizon, dim_u))
    state = initial_state
    for t in range(horizon):
        action = global_policy.predict(state, t=t)
        xu = torch.cat((state, action))[None, :]
        x_ = global_dynamics.predict(xu)
        xs[t, :] = state
        us[t, :] = action
        state = x_[0, :]
    environment.plot(xs, us)  # env.plot does not use env, it only plots
    plt.suptitle(f"{policy_type} policy vs {dyn_model_type} dynamics")
    plt.savefig(
        results_dir / f"{policy_type}_vs_{dyn_model_type}_{i_iter}.png", dpi=150
    )

    ### plot data space coverage ###
    if plot_data:
        cols = ["theta", "theta_dot"]  # TODO useful to also plot actions?
        # train data
        xs = dyn_train_buffer.xs[:, :dim_x]  # only states
        df = pd.DataFrame()
        df[cols] = np.array(xs.cpu())
        df["traj_id"] = df.index // horizon
        g = sns.PairGrid(df, hue="traj_id")
        # g = sns.PairGrid(df)
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.fig.suptitle(f"train data ({df.shape[0] // horizon} episodes)", y=1.01)
        g.savefig(results_dir / "train_data.png", dpi=150)
        # test data
        xs = dyn_test_buffer.xs[:, :dim_x]  # only states
        df = pd.DataFrame()
        df[cols] = np.array(xs.cpu())
        df["traj_id"] = df.index // horizon
        g = sns.PairGrid(df, hue="traj_id")
        # g = sns.PairGrid(df)
        g.map_diag(sns.histplot, hue=None)
        g.map_offdiag(plt.plot)
        g.fig.suptitle(f"test data ({df.shape[0] // horizon} episodes)", y=1.01)
        g.savefig(results_dir / "test_data.png", dpi=150)

    # plot training loss
    def scaled_xaxis(y_points, n_on_axis):
        return np.arange(len(y_points)) / len(y_points) * n_on_axis

    if dyn_model_type != "env":
        fig_loss_dyn, ax_loss_dyn = plt.subplots()
        x_train_loss_dyn = scaled_xaxis(dyn_loss_trace, n_iter * n_epochs_dyn)
        ax_loss_dyn.plot(x_train_loss_dyn, dyn_loss_trace, c="k", label="train loss")
        x_test_loss_dyn = scaled_xaxis(dyn_test_loss_trace, n_iter * n_epochs_dyn)
        ax_loss_dyn.plot(x_test_loss_dyn, dyn_test_loss_trace, c="g", label="test loss")
        if dyn_loss_trace[0] > 1 and dyn_loss_trace[-1] < 0.1:
            ax_loss_dyn.set_yscale("symlog")
        ax_loss_dyn.set_xlabel("epochs")
        ax_loss_dyn.set_ylabel("loss")
        ax_loss_dyn.set_title(
            f"DYN {dyn_model_type} loss "
            f"({int(dyn_train_buffer.size/horizon)} episodes, lr={lr_dyn:.0e})"
        )
        ax_loss_dyn.legend()
        plt.savefig(results_dir / "dyn_loss.png", dpi=150)
        # TODO plot policy train loss

    if plotting:
        plt.show()

    logger.strong_line()
    logger.info(f"Seed: {seed} - Took {time.time()-time_begin:.2f} seconds")
    logger.info(f"Logs in {results_dir}")
    logger.finish()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)
