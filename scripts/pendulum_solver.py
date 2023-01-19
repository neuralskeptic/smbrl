import math
import pdb
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, partialmethod
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import torch
from pytorch_minimize.optim import MinimizeWrapper
from torch.autograd.functional import hessian, jacobian


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
            self.mean[indices], self.covariance[indices, indices], None, None, None
        )

    def full_joint(self, reverse=True):
        """Convert Markovian structure into the joint multivariate Gaussian and forget history."""
        if reverse:
            mean = torch.cat((self.previous_mean, self.mean), axis=0)
            covariance = torch.cat(
                (
                    torch.cat(
                        (self.previous_covariance, self.cross_covariance.T), axis=1
                    ),
                    (torch.cat((self.cross_covariance, self.covariance), axis=1)),
                ),
                axis=0,
            )
        else:
            mean = torch.cat((self.mean, self.previous_mean), axis=0)
            covariance = torch.cat(
                (
                    torch.cat((self.covariance, self.cross_covariance), axis=1),
                    (
                        torch.cat(
                            (self.cross_covariance.T, self.previous_covariance), axis=1
                        )
                    ),
                ),
                axis=0,
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
            self.cross_covariance.T,
            self.mean,
            self.covariance,
        )


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
        try:
            sqrt = torch.linalg.cholesky(sigma_x)
        except torch.linalg.LinAlgError:
            print(
                f"singular covariance: {torch.linalg.eigvalsh(sigma_x)}, state: {mu_x}"
            )
            sqrt = torch.linalg.cholesky(torch.diag(torch.diag(sigma_x)))
        scale = self.sf * sqrt
        return mu_x[None, :] + self.base_pts @ scale.T

    def __call__(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> MultivariateGaussian:
        assert isinstance(distribution, MultivariateGaussian)
        x_pts = self.get_x_pts(distribution.mean, distribution.covariance)
        self.y_pts, self.m_y, self.sig_y, self.sig_xy = self.forward_pts(
            function, distribution.mean, x_pts
        )
        return MultivariateGaussian(
            self.m_y,
            self.sig_y,
            self.sig_xy.T,
            distribution.mean,
            distribution.covariance,
        )

    def forward_pts(
        self, f: Callable, mu_x: torch.Tensor, x_pts: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        y_pts, x_pts = f(x_pts)
        mu_y = torch.einsum("b,bi->i", self.weights_mu, y_pts)
        diff_x = x_pts - mu_x[None, :]
        diff_y = y_pts - mu_y[None, :]
        sigma_y = torch.einsum(
            "b,bi,bj->ij", self.weights_sig, diff_y, diff_y
        ) + 1e-7 * torch.eye(y_pts.shape[1])
        sigma_xy = torch.einsum("b,bi,bj->ij", self.weights_sig, diff_x, diff_y)
        return y_pts, mu_y, sigma_y, sigma_xy

    def mean(
        self, function: Callable, distribution: MultivariateGaussian
    ) -> torch.float32:
        return torch.einsum(
            "b,b->",
            self.weights_mu,
            function(self.get_x_pts(distribution.mean, distribution.covariance))[0],
        )


class QuadratureGaussianInference(QuadratureInference):
    """For a function that returns a mean and covariance."""

    def forward_pts(
        self, f: Callable, mu_x: torch.Tensor, x_pts: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        y_pts, sigma_pts = f(x_pts)
        mu_y = torch.einsum("b,bi->i", self.weights_mu, y_pts)
        diff_x = x_pts - mu_x[None, :]
        diff_y = y_pts - mu_y[None, :]
        sigma_y = torch.einsum(
            "b,bi,bj->ij", self.weights_sig, diff_y, diff_y
        ) + torch.einsum("b,bij->ij", self.weights_mu, sigma_pts)
        sigma_xy = torch.einsum("b,bi,bj->ij", self.weights_sig, diff_x, diff_y)
        return y_pts, mu_y, sigma_y, sigma_xy


class QuadratureImportanceSamplingInnovation(QuadratureInference):
    def __call__(
        self,
        function: Callable,
        inverse_temperature: float,
        distribution: MultivariateGaussian,
    ) -> MultivariateGaussian:
        assert isinstance(distribution, MultivariateGaussian)
        x_pts = self.get_x_pts(distribution.mean, distribution.covariance)
        f_pts, self.x_pts = function(x_pts)
        log_weights = -inverse_temperature * f_pts + torch.log(self.weights_mu)
        log_weights -= torch.logsumexp(log_weights, dim=0)
        weights = torch.exp(log_weights)
        mu_x = torch.einsum("b, bi -> i", weights, self.x_pts)
        diff = self.x_pts - mu_x[None, :]
        sigma_x = torch.einsum("b, bi, bj -> ij", weights, diff, diff)
        sigma_x = 0.5 * (sigma_x + sigma_x.T)
        return MultivariateGaussian(mu_x, sigma_x, None, None, None)


def linear_gaussian_smoothing(
    current_prior: MultivariateGaussian,
    predicted_prior: MultivariateGaussian,
    future_posterior: MultivariateGaussian,
) -> MultivariateGaussian:
    """.
    See https://vismod.media.mit.edu/tech-reports/TR-531.pdf, Section 3.2
    """
    J = torch.linalg.solve(
        predicted_prior.covariance, predicted_prior.cross_covariance
    ).T
    mean = current_prior.mean + J @ (future_posterior.mean - predicted_prior.mean)
    covariance = (
        current_prior.covariance
        + J @ (future_posterior.covariance - predicted_prior.covariance) @ J.T
    )
    return MultivariateGaussian(
        mean,
        covariance,
        J @ future_posterior.covariance,
        future_posterior.mean,
        future_posterior.covariance,
    )


class Policy(ABC):
    @abstractmethod
    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        pass

    @abstractmethod
    def update_from_distribution(self, distribution, *args, **kwargs):
        pass


class TimeVaryingLinearGaussian(Policy):
    def __init__(
        self, horizon: int, dim_x: int, dim_u: int, action_covariance: torch.Tensor
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

        copy = deepcopy(self)
        copy.k_opt = self.k_actual
        copy.K_opt = self.K_actual
        return copy

    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Assumes batch input."""
        return self.k_opt[t, :] + x @ self.K_opt[t, :, :].T, torch.tile(
            self.sigma[t, :, :], (x.shape[0], 1, 1)
        )

    def predict(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.k_actual[t, :] + x @ self.K_actual[t, :, :].T

    def sample(self, t: int, x: torch.Tensor) -> torch.Tensor:
        mu = self.k[t, :] + self.K[t, :, :] @ x
        eps = torch.randn(self.dim_u)
        return mu + self.chol[t, :, :] @ eps

    def update_from_distribution(self, distribution, *args, **kwargs):
        assert len(distribution) == self.horizon
        if isinstance(distribution[0], MultivariateGaussian):
            for t, dist in enumerate(distribution):
                x, u = dist.marginalize(slice(0, self.dim_x)), dist.marginalize(
                    slice(self.dim_x, self.dim_x + self.dim_u)
                )
                K = torch.linalg.solve(
                    x.covariance, dist.covariance[: self.dim_x, self.dim_x :]
                ).T
                self.K_actual[t, ...] = K
                self.k_actual[t, ...] = u.mean - self.K_actual[t, ...] @ x.mean
                self.k_opt[t, ...] = u.mean - self.K_opt[t, ...] @ x.mean
                self.sigma[t, ...] = u.covariance
                self.chol[t, ...] = torch.linalg.cholesky(self.sigma[t, ...])
        else:
            raise ValueError("Cannot update from this distribution!")


class Pendulum(object):

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
        x, u = xu[:, :2], xu[:, 2]
        u = torch.clip(u, -self.u_mx, self.u_mx)
        th_dot_dot = -3.0 * g / (2 * l) * torch.sin(x[:, 0] + torch.pi) - d * x[:, 1]
        th_dot_dot += 3.0 / (m * l**2) * u
        x_dot = x[:, 1] + th_dot_dot * dt
        x_pos = x[:, 0] + x_dot * dt

        x2 = torch.stack((x_pos, x_dot), dim=1)
        xu[:, 2] = u
        return x2, xu

    def run(self, initial_state, policy, horizon):
        xs = torch.zeros((horizon, self.dim_x))
        us = torch.zeros((horizon, self.dim_u))
        state = initial_state
        for t in range(horizon):
            action = policy.predict(state, t)
            xu = torch.cat((state, action))[None, :]
            x_, _ = self.__call__(xu)
            xs[t, :] = state
            us[t, :] = action
            state = x_[0, :]
        return xs, us

    def plot(self, xs, us):
        fig, axs = plt.subplots(self.dim_x + self.dim_u, figsize=(12, 9))
        for i in range(self.dim_x):
            axs[i].plot(xs[:, i])
            axs[i].set_ylabel(f"x{i}")
        for i in range(self.dim_u):
            j = self.dim_x + i
            axs[j].plot(us[:, i])
            axs[j].plot(self.u_mx * torch.ones_like(us[:, i]), "k--")
            axs[j].plot(-self.u_mx * torch.ones_like(us[:, i]), "k--")
            axs[j].set_ylabel(f"u{i}")
        axs[-1].set_xlabel("timestep")


def plot_gp(axis, mean, variance):
    axis.plot(mean, "b")
    sqrt = torch.sqrt(variance)
    upper, lower = mean + sqrt, mean - sqrt
    axis.fill_between(
        range(mean.shape[0]), upper, lower, where=upper >= lower, color="b", alpha=0.3
    )


def plot_trajectory_distribution(list_of_distributions, title=""):
    means = torch.stack(tuple(d.mean for d in list_of_distributions), dim=0)
    covariances = torch.stack(tuple(d.covariance for d in list_of_distributions), dim=0)
    n_plots = means.shape[1]
    fig, axs = plt.subplots(n_plots)
    axs[0].set_title(title)
    for i, ax in enumerate(axs):
        plot_gp(ax, means[:, i], covariances[:, i, i])
    return fig, axs
    # plt.show()  # turn on debugging


class PseudoPosteriorSolver(object):

    metrics: Dict

    def __init__(
        self,
        env: Callable,
        cost: Callable,
        horizon: int,
        initial_state_distribution: Distribution,
        policy_prior: Policy,
        approximate_inference_dynamics,
        approximate_inference_policy,
        approximate_inference_cost,
        approximate_inference_smoothing,
        update_temperature_strategy,
    ):
        self.env = env
        self.dim_x, self.dim_u = env.dim_x, env.dim_x
        self.cost = cost
        self.horizon = horizon
        self.initial_state = initial_state_distribution
        self.policy_prior = policy_prior
        self.alpha = 0.0
        self.approximate_inference_dynamics = approximate_inference_dynamics
        self.approximate_inference_policy = approximate_inference_policy
        self.approximate_inference_cost = approximate_inference_cost
        self.approximate_inference_smoothing = approximate_inference_smoothing
        self.update_temperature_strategy = update_temperature_strategy

    def forward_pass(
        self,
        env: Callable,
        cost: Callable,
        policy: Policy,
        initial_state: Distribution,
        alpha: float,
    ) -> (List[Distribution], Distribution):
        state = initial_state
        state_action_dist = []
        future_state_dist = []
        for t in range(self.horizon):
            policy_ = partial(policy.__call__, t=t)
            action = self.approximate_inference_policy(policy_, state)
            state_action_policy = action.full_joint(reverse=True)
            state_action_cost = self.approximate_inference_cost(
                cost, alpha, state_action_policy
            )
            state_action_dist += [state_action_cost]
            # update state_action_cost with dynamics correlations
            next_state = self.approximate_inference_dynamics(env, state_action_cost)
            future_state_dist += [next_state]
            state = next_state
        return state_action_dist, future_state_dist, state

    def backward_pass(
        self,
        forward_state_action_distribution,
        predicted_state_distribution,
        terminal_state_distribution,
    ):
        dist = []
        future_state_posterior = terminal_state_distribution
        for current_state_action_prior, predicted_state_prior in zip(
            reversed(forward_state_action_distribution),
            reversed(predicted_state_distribution),
        ):
            state_action_posterior = self.approximate_inference_smoothing(
                current_state_action_prior,
                predicted_state_prior,
                future_state_posterior,
            )
            future_state_posterior = state_action_posterior.marginalize(
                slice(0, self.dim_x)
            )  # pass the state posterior to previous timestep
            dist += [state_action_posterior]
        return list(reversed(dist))

    def __call__(self, n_iteration: int):
        initial_alpha = 0.0
        self.init_metrics()
        policy = self.policy_prior
        (
            forward_state_action_prior,
            next_state_state_action_prior,
            _,
        ) = self.forward_pass(
            self.env, self.cost, policy, self.initial_state, alpha=initial_alpha
        )
        forward_distribution = [
            dist.reverse() for dist in next_state_state_action_prior
        ]
        self.alpha = self.update_temperature_strategy(
            self.cost,
            forward_distribution,
            forward_distribution,
            current_alpha=initial_alpha,
        )
        # plot_trajectory_distribution(forward_state_action_prior, f"init")
        self.compute_metrics(
            forward_state_action_prior, forward_state_action_prior, initial_alpha
        )
        print(self.alpha)
        for i in range(n_iteration):
            print(f"{i} {self.alpha:.2f}")
            (
                forward_state_action_prior,
                predicted_state_prior,
                terminal_state,
            ) = self.forward_pass(
                self.env, self.cost, policy, self.initial_state, self.alpha
            )
            # plot_trajectory_distribution(forward_state_action_prior, f"filter{i}")
            state_action_posterior = self.backward_pass(
                forward_state_action_prior, predicted_state_prior, terminal_state
            )
            state_action_policy, _, _ = self.forward_pass(
                self.env, self.cost, policy.actual(), self.initial_state, alpha=0.0
            )
            self.compute_metrics(
                state_action_posterior, state_action_policy, self.alpha
            )
            policy.update_from_distribution(state_action_posterior, state_action_policy)
            self.alpha = self.update_temperature_strategy(
                self.cost,
                state_action_posterior,
                state_action_policy,
                current_alpha=initial_alpha,
            )
            # self.alpha = self.update_temperature_strategy(self.cost, [dist.reverse() for dist in next_state_state_action_prior], current_alpha=initial_alpha)
            # plot_trajectory_distribution(state_action_posterior, f"smooth{i}")
            # plt.show()
            # if converged(metrics):
            #   break
        # plot_trajectory_distribution(forward_state_action_prior, f"filter{i}")
        plot_trajectory_distribution(state_action_posterior, "posterior")
        return policy

    def compute_metrics(self, posterior_distribution, policy_distribution, alpha):
        posterior_cost = self.cost(
            torch.stack(tuple(d.mean for d in posterior_distribution), dim=0)
        )[0].sum()
        policy_cost = self.cost(
            torch.stack(tuple(d.mean for d in policy_distribution), dim=0)
        )[0].sum()
        self.metrics["posterior_cost"] += [posterior_cost]
        self.metrics["policy_cost"] += [policy_cost]
        self.metrics["alpha"] += [alpha]

    def init_metrics(self):
        self.metrics = {
            "posterior_cost": [],
            "policy_cost": [],
            "alpha": [],
        }

    def plot_metrics(self):
        fix, axs = plt.subplots(len(self.metrics))
        for i, (k, v) in enumerate(self.metrics.items()):
            axs[i].plot(v)
            axs[i].set_ylabel(k)


class TemperatureStrategy(ABC):

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
        traj_costs, _ = cost(traj_points)
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
        state_action_distribution,
        state_action_policy,
        current_alpha: float,
    ):
        list_of_weights, list_of_costs = zip(
            *[
                (
                    self.inference.weights_mu,
                    cost(self.inference.get_x_pts(dist.mean, dist.covariance))[0],
                )
                for dist in state_action_policy
            ]
        )
        traj_weights, traj_costs = map(torch.cat, [list_of_weights, list_of_costs])
        sup_cost = sum([torch.max(costs_t) for costs_t in list_of_costs])
        expected_cost = torch.einsum("b,b->", traj_weights, traj_costs)
        return expected_cost / sup_cost


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


if __name__ == "__main__":
    torch.set_printoptions(precision=7)
    torch.set_default_dtype(torch.float64)
    task_horizon = 200
    environment = Pendulum()
    initial_state = torch.Tensor([torch.pi, 0.0])

    def state_action_cost(xu):
        # swing-up from \theta = \pi -> 0
        return (torch.cos(xu[:, 0]) - 1.0) ** 2 + 1e-2 * xu[:, 1] ** 2 + 1e-2 * xu[
            :, 2
        ] ** 2, xu

    quad_params = CubatureQuadrature(1, 0, 0)
    gh_params = GaussHermiteQuadrature(degree=3)
    initial_state_distribution = MultivariateGaussian(
        initial_state, 1e-6 * torch.eye(environment.dim_x), None, None, None
    )
    fix, axs = plt.subplots(3)
    # for alpha in [1e-1, 1, 10, 100, 1000]:
    # for kl_bound in [1e-4, 1e-3, 1e-2, 1e-1, 1]:
    for _ in [None]:
        policy = TimeVaryingLinearGaussian(
            task_horizon,
            environment.dim_x,
            environment.dim_u,
            action_covariance=0.2 * torch.eye(environment.dim_u),
        )
        agent = PseudoPosteriorSolver(
            environment,
            state_action_cost,
            task_horizon,
            initial_state_distribution,
            policy_prior=policy,
            approximate_inference_dynamics=QuadratureInference(
                environment.dim_xu, quad_params
            ),
            approximate_inference_policy=QuadratureGaussianInference(
                environment.dim_x, quad_params
            ),
            approximate_inference_cost=QuadratureImportanceSamplingInnovation(
                environment.dim_xu, gh_params
            ),
            # approximate_inference_cost=LinearizationInnovation(),
            approximate_inference_smoothing=linear_gaussian_smoothing,
            # update_temperature_strategy=Constant(alpha),
            # update_temperature_strategy=MaximumLikelihood(QuadratureInference(environment.dim_xu, quad_params)),
            # update_temperature_strategy=QuadraticModel(QuadratureInference(environment.dim_xu, quad_params)),
            # update_temperature_strategy=KullbackLeiblerDivergence(QuadratureInference(environment.dim_xu, gh_params), epsilon=kl_bound),
            # update_temperature_strategy=Annealing(1e-2, 5e1, 15+1),
            update_temperature_strategy=PolyakStepSize(
                QuadratureInference(environment.dim_xu, quad_params)
            ),
        )
        policy = agent(n_iteration=10)

        for i, (k, v) in enumerate(agent.metrics.items()):
            # axs[i].plot(v, label=f"$\epsilon={kl_bound}$")
            # axs[i].plot(v, label=f"$\\alpha={alpha}$")
            axs[i].plot(v, label=f"$Polyak$")
            # axs[i].plot(v, label=f"Maximum Likelihood")
            # axs[i].plot(v, label="Quadratic Model")
            # axs[i].plot(v, label="Annealing")
            axs[i].set_ylabel(k)
    for ax in axs:
        ax.legend()
    # agent.plot_metrics()
    trajectory = environment.run(initial_state, policy, task_horizon)
    environment.plot(*trajectory)
    plt.show()
